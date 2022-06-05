#include <iostream>
#include "tbb/tbbmalloc_proxy.h"
#include "simd_vdb_poisson.h"
#include "simd_viscosity3d.h"
#include "openvdb/tools/LevelSetSphere.h"
#include "openvdb/tools/MeshToVolume.h"
#include "openvdb/tools/Morphology.h"
#include "openvdb/tools/ParticlesToLevelSet.h"
#include "YamlSingleton.h"
#include "FLIP_vdb.h"
#include "pcgsolver/pcg_solver.h"
#include "Tracy.hpp"
#include <filesystem>
#include "Timer.h"

#include "amgcl/backend/builtin.hpp"
#include "amgcl/adapter/crs_tuple.hpp"
#include "amgcl/make_solver.hpp"
#include "amgcl/amg.hpp"
#include "amgcl/coarsening/smoothed_aggregation.hpp"
#include "amgcl/coarsening/aggregation.hpp"
#include "amgcl/coarsening/ruge_stuben.hpp"
#include "amgcl/relaxation/spai0.hpp"
#include "amgcl/relaxation/spai1.hpp"
#include "amgcl/relaxation/damped_jacobi.hpp"
#include "amgcl/solver/bicgstab.hpp"
#include "amgcl/solver/cg.hpp"


void unit_test_PCG(openvdb::FloatGrid::Ptr in_viscosity,
	openvdb::FloatGrid::Ptr in_liquid_sdf,
	openvdb::FloatGrid::Ptr in_solid_sdf) {
	packed_FloatGrid3 liquid_velocity;

	//set the initial velocity to follow the liquid sdf, but with randomness
	for (int i = 0; i < 3; i++) {
		liquid_velocity.v[i]->setTree(
			std::make_shared<openvdb::FloatTree>(in_liquid_sdf->tree(), 0.f, openvdb::TopologyCopy()));
		auto treeman = openvdb::tree::LeafManager<openvdb::FloatTree>(liquid_velocity.v[i]->tree());
		float nleaf = treeman.leafCount();
		auto random_setter = [&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index leafpos) {
			std::random_device device;
			std::mt19937 generator(/*seed=*/device());
			std::uniform_real_distribution<> distribution(-0.5, 0.5);
			for (auto iter = leaf.beginValueAll(); iter; ++iter) {
				iter.setValue(iter.getValue() + distribution(generator) + (iter.offset() - 256.f) * (1.0f / 512.0f) + (leafpos - nleaf / 2) / nleaf);
				//iter.setValue(0);
				iter.setValueOn();
			}
		};
		treeman.foreach(random_setter);
	}

	//liquid_velocity.v[0]->tree().setValue(openvdb::Coord(0, 0, 0), 1000);
	//openvdb::io::File("liquid_velocity.vdb").write({ liquid_velocity.v[0], liquid_velocity.v[1], liquid_velocity.v[2] });
	openvdb::Vec3fGrid::Ptr solid_velocity = openvdb::Vec3fGrid::create(openvdb::Vec3f(1.f));

	simd_uaamg::simd_viscosity3d simdsolver(in_viscosity,
		in_liquid_sdf,
		in_solid_sdf,
		liquid_velocity,
		solid_velocity, 1.0f, 1.0f);

	packed_FloatGrid3 result = simdsolver.m_matrix_levels[0]->get_zero_vec();

	printf("solve\n");
	simdsolver.pcg_solve(result, 1e-7);
	printf("solvedone\n");

	//openvdb::io::File("result.vdb").write({ result.v[0], result.v[1], result.v[2] });
	//openvdb::io::File("rhs.vdb").write({ simdsolver.m_rhs.v[0], simdsolver.m_rhs.v[1], simdsolver.m_rhs.v[2] });
}

template <typename GridType>
static void
sopFillSDF(GridType& grid, int dummy)
{
	typename GridType::Accessor         access = grid.getAccessor();
	typedef typename GridType::ValueType ValueT;

	ValueT              value;
	ValueT              background = grid.background();

	for (typename GridType::ValueOffCIter
		iter = grid.cbeginValueOff(); iter; ++iter)
	{

		openvdb::CoordBBox bbox = iter.getBoundingBox();

		// Assuming the SDF is at all well-formed, any crossing
		// of sign must have a crossing of inactive->active.
		openvdb::Coord coord(bbox.min().x(), bbox.min().y(), bbox.min().z());

		// We do not care about the active state as it is hopefully inactive
		access.probeValue(coord, value);

		if (value < 0)
		{
			// Fill the region to negative background.
			grid.fill(bbox, -background, /*active=*/true);
		}
	}
}


//create a liquid cube with dx and cube size
//The solid_sdf is a container to hold the liquid sdf
//it is slightly higher than the liquid cube
void compact_poisson_scene(openvdb::FloatGrid::Ptr in_out_liquid_sdf, openvdb::FloatGrid::Ptr in_out_solid_sdf, const float in_dx, openvdb::Vec3f cube_size) {
	auto voxel_center_transform = openvdb::math::Transform::createLinearTransform(in_dx);
	auto voxel_corner_transform = voxel_center_transform->copy();
	auto halfvoxel = openvdb::Vec3f(0.5f * in_dx);
	voxel_corner_transform->postTranslate(-halfvoxel);

	//Liquid part
	auto liquid_box_sdf = openvdb::tools::createLevelSetBox<openvdb::FloatGrid>(
		openvdb::BBoxd(openvdb::Vec3f(0.f), cube_size), *voxel_center_transform);
	sopFillSDF(*liquid_box_sdf, 0);
	liquid_box_sdf->tree().voxelizeActiveTiles();
	in_out_liquid_sdf->setName("liquid_sdf");
	in_out_liquid_sdf->setGridClass(openvdb::GridClass::GRID_LEVEL_SET);
	in_out_liquid_sdf->setTransform(voxel_center_transform);
	in_out_liquid_sdf->setTree(liquid_box_sdf->treePtr());

	//Container part
	auto container_size = cube_size;
	container_size.y() *= 1.5f;

	auto container_air_sdf = openvdb::tools::createLevelSetBox<openvdb::FloatGrid>(
		openvdb::BBoxd(halfvoxel, container_size - halfvoxel), *voxel_corner_transform);

	auto  container_solid_sdf = openvdb::FloatGrid::create(container_air_sdf->background());

	for (auto iter = container_air_sdf->tree().beginLeaf(); iter; ++iter) {
		container_solid_sdf->tree().addLeaf(new openvdb::FloatTree::LeafNodeType(*iter));
	}

	auto invert_sign = [&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index) {
		for (auto iter = leaf.beginValueAll(); iter; ++iter) {
			if (iter.isValueOn()) {
				iter.setValue(-iter.getValue());
			}
			else {
				iter.setValue(container_air_sdf->background());
			}
		}
	};
	auto sdfman = openvdb::tree::LeafManager<openvdb::FloatTree>(container_solid_sdf->tree());
	sdfman.foreach(invert_sign);

	in_out_solid_sdf->setName("solid_sdf");
	in_out_solid_sdf->setGridClass(openvdb::GridClass::GRID_LEVEL_SET);
	in_out_solid_sdf->setTransform(voxel_corner_transform);
	in_out_solid_sdf->setTree(container_solid_sdf->treePtr());
}


namespace {
struct ParticleList
{
	using PosType = openvdb::Vec3R;
	// Return the total number of particles in the list.
	// Always required!
	size_t size() const { return m_particles.size(); };

	// Get the world-space position of the nth particle.
	// Required by rasterizeSpheres().
	void getPos(size_t n, openvdb::Vec3R& xyz) const {
		xyz = m_particles[n];
	};

	// Get the world-space position and radius of the nth particle.
	// Required by rasterizeSpheres().
	void getPosRad(size_t n, openvdb::Vec3R& xyz, openvdb::Real& radius) const {
		xyz = m_particles[n];
		radius = m_radius;
	};;

	// Get the world-space position, radius and velocity of the nth particle.
	// Required by rasterizeTrails().
	void getPosRadVel(size_t n, openvdb::Vec3R& xyz, openvdb::Real& radius, openvdb::Vec3R& velocity) const;

	// Get the value of the nth particle's user-defined attribute (of type @c AttributeType).
	// Required only if attribute transfer is enabled in ParticlesToLevelSet.
	void getAtt(size_t n, float& att) const;

	std::vector<openvdb::Vec3R> m_particles;
	float m_radius;
};
}
void sparse_poisson_scene(openvdb::FloatGrid::Ptr in_out_liquid_sdf, openvdb::FloatGrid::Ptr in_out_solid_sdf, const float in_dx, openvdb::Vec3f cube_size) {
	//fill the cube with randomly-positioned balls 
	int nballs = YamlSingleton::get()["nballs"].as<int>();

	float cube_volume = cube_size.x() * cube_size.y() * cube_size.z();
	float resizer = YamlSingleton::get()["resizer"].as<float>();
	float ball_radius = resizer  * std::cbrt(cube_volume * 3.0f / (4.0f * 3.14f) / nballs);

	ParticleList particles;
	particles.m_particles.reserve(nballs);
	particles.m_radius = ball_radius;

	std::mt19937 generator(/*seed=*/1);
	std::uniform_real_distribution<> distribution(0, 1);
	for (int i = 0; i < nballs; i++) {
		openvdb::Vec3R new_pos{ cube_size };
		for (int c = 0; c < 3; c++) {
			new_pos[c] *= distribution(generator);
		}
		//printf("%f %f %f\n", new_pos[0], new_pos[1], new_pos[2]);
		particles.m_particles.push_back(new_pos);
	}

	openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid> p2ls(*in_out_liquid_sdf, nullptr);
	p2ls.setRmax(2 * int(1 / in_dx));
	p2ls.rasterizeSpheres(particles);
	openvdb::tools::pruneLevelSet((*in_out_liquid_sdf).tree());

	//openvdb::tools::particlesToSdf(particles, *in_out_liquid_sdf);
	sopFillSDF(*in_out_liquid_sdf, 0);
	in_out_liquid_sdf->tree().voxelizeActiveTiles();
}

void viscosity_scene() {

}

float sparsity_fraction_dof(openvdb::Int32Tree& in_dof_tree) {
	int total_voxels = in_dof_tree.leafCount() * 512;
	std::atomic<int> active_voxel_count{ 0 };
	std::atomic<int> total_lane_voxels{ 0 };
	auto active_voxel_counter = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index) {
		int inside_count = 0;
		int inside_lane_voxel_count = 0;
		for (uint32_t laneoffset = 0; laneoffset < 512; laneoffset += 8) {
			const uint8_t lanemask = leaf.getValueMask().getWord<uint8_t>(laneoffset / 8);
			if (lanemask == uint8_t(0)) {
				//there is no diagonal entry in this lane
				continue;
			}
			inside_lane_voxel_count += 8;
			for (int bit = 0; bit < 8; bit += 1) {
				if (0 != ((lanemask) & (1 << bit))) {
					inside_count++;
				}
			}
		}
		total_lane_voxels += inside_lane_voxel_count;
		active_voxel_count += inside_count;
	};
	openvdb::tree::LeafManager<openvdb::Int32Tree> dofman(in_dof_tree);
	dofman.foreach(active_voxel_counter);

	return float(active_voxel_count) / float(total_lane_voxels + 1);
}

void solve_poisson_Eigen_diagonalPCG(
	openvdb::FloatGrid::Ptr rhs,
	openvdb::FloatGrid::Ptr initial_guess,
	simd_uaamg::LaplacianWithLevel::Ptr level0) {
	std::vector<float> vector_rhs, vector_x;

	CSim::TimerMan::timer("Benchmark/solve/cvtvector").start();
	level0->gridToVector(vector_rhs, rhs);
	level0->gridToVector(vector_x, initial_guess);
	CSim::TimerMan::timer("Benchmark/solve/cvtvector").stop();

	//right hand side, initial guess
	Eigen::VectorXf eigen_rhs, eigen_x;
	eigen_rhs = Eigen::Map<Eigen::VectorXf>(vector_rhs.data(), vector_rhs.size());
	eigen_x = Eigen::Map<Eigen::VectorXf>(vector_x.data(), vector_x.size());

	//left hand side matrix
	std::vector<Eigen::Triplet<float>> triplets;
	CSim::TimerMan::timer("Benchmark/solve/gettriplets").start();
	level0->getTriplets(triplets);
	CSim::TimerMan::timer("Benchmark/solve/gettriplets").stop();
	CSim::TimerMan::timer("Benchmark/solve/build_explicit_matrix").start();
	Eigen::SparseMatrix<float, Eigen::RowMajor> eigen_lhs;
	eigen_lhs.resize(eigen_rhs.size(), eigen_rhs.size());
	eigen_lhs.setFromTriplets(triplets.begin(), triplets.end());
	CSim::TimerMan::timer("Benchmark/solve/build_explicit_matrix").stop();

	//the solver
	CSim::TimerMan::timer("Benchmark/solve/CG_init").start();
	Eigen::ConjugateGradient<Eigen::SparseMatrix<float, Eigen::RowMajor>, Eigen::Lower | Eigen::Upper, Eigen::DiagonalPreconditioner<float>> diagonal_PCG(eigen_lhs);
	CSim::TimerMan::timer("Benchmark/solve/CG_init").stop();
	printf("start solving DPCG... with %d threads\n", Eigen::nbThreads());
	std::cout << "Eigen DPCG tolerance: " << diagonal_PCG.tolerance() << std::endl;
	CSim::TimerMan::timer("Benchmark/solve/Eigen_DPCG").start();
	eigen_x = diagonal_PCG.solveWithGuess(eigen_rhs, eigen_x);
	CSim::TimerMan::timer("Benchmark/solve/Eigen_DPCG").stop();
	CSim::TimerMan::timer("Benchmark/solve/cvtgrid").start();
	std::cout << "Eigen DPCG solve iteration: " << diagonal_PCG.iterations() << std::endl;
	std::cout << "Eigen DPCG residual: " << diagonal_PCG.error() << std::endl;
	std::copy(eigen_x.data(), eigen_x.data() + eigen_x.size(), vector_x.data());
	level0->vectorToGrid(initial_guess, vector_x);
	CSim::TimerMan::timer("Benchmark/solve/cvtgrid").stop();
}

void solve_poisson_Eigen_ICPCG(
	openvdb::FloatGrid::Ptr rhs,
	openvdb::FloatGrid::Ptr initial_guess,
	simd_uaamg::LaplacianWithLevel::Ptr level0) {
	std::vector<float> vector_rhs, vector_x;

	CSim::TimerMan::timer("Benchmark/solve/cvtvector").start();
	level0->gridToVector(vector_rhs, rhs);
	level0->gridToVector(vector_x, initial_guess);
	CSim::TimerMan::timer("Benchmark/solve/cvtvector").stop();

	//right hand side, initial guess
	Eigen::VectorXf eigen_rhs, eigen_x;
	eigen_rhs = Eigen::Map<Eigen::VectorXf>(vector_rhs.data(), vector_rhs.size());
	eigen_x = Eigen::Map<Eigen::VectorXf>(vector_x.data(), vector_x.size());

	//left hand side matrix
	std::vector<Eigen::Triplet<float>> triplets;
	CSim::TimerMan::timer("Benchmark/solve/gettriplets").start();
	level0->getTriplets(triplets);
	CSim::TimerMan::timer("Benchmark/solve/gettriplets").stop();
	CSim::TimerMan::timer("Benchmark/solve/build_explicit_matrix").start();
	Eigen::SparseMatrix<float, Eigen::RowMajor> eigen_lhs;
	eigen_lhs.resize(eigen_rhs.size(), eigen_rhs.size());
	eigen_lhs.setFromTriplets(triplets.begin(), triplets.end());
	CSim::TimerMan::timer("Benchmark/solve/build_explicit_matrix").stop();

	//the solver
	CSim::TimerMan::timer("Benchmark/solve/CG_init").start();
	Eigen::ConjugateGradient<Eigen::SparseMatrix<float, Eigen::RowMajor>, Eigen::Lower | Eigen::Upper, Eigen::IncompleteCholesky<float>> IC_PCG(eigen_lhs);
	CSim::TimerMan::timer("Benchmark/solve/CG_init").stop();
	printf("start solving ICPCG... with %d threads\n", Eigen::nbThreads());
	std::cout << "Eigen ICPCG tolerance: " << IC_PCG.tolerance() << std::endl;
	CSim::TimerMan::timer("Benchmark/solve/Eigen_ICPCG").start();
	eigen_x = IC_PCG.solveWithGuess(eigen_rhs, eigen_x);
	CSim::TimerMan::timer("Benchmark/solve/Eigen_ICPCG").stop();
	std::cout << "Eigen ICPCG solve iteration: " << IC_PCG.iterations() << std::endl;
	std::cout << "Eigen ICPCG residual: " << IC_PCG.error() << std::endl;
	CSim::TimerMan::timer("Benchmark/solve/cvtgrid").start();
	std::copy(eigen_x.data(), eigen_x.data() + eigen_x.size(), vector_x.data());
	
	level0->vectorToGrid(initial_guess, vector_x);
	CSim::TimerMan::timer("Benchmark/solve/cvtgrid").stop();
}


void solve_poisson_Batty_ICPCG(
	openvdb::FloatGrid::Ptr rhs,
	openvdb::FloatGrid::Ptr initial_guess,
	simd_uaamg::LaplacianWithLevel::Ptr level0) {
	std::vector<float> vector_rhs, vector_x;

	CSim::TimerMan::timer("Benchmark/solve/cvtvector").start();
	level0->gridToVector(vector_rhs, rhs);
	level0->gridToVector(vector_x, initial_guess);
	CSim::TimerMan::timer("Benchmark/solve/cvtvector").stop();
	//left hand side matrix
	std::vector<Eigen::Triplet<float>> triplets;
	CSim::TimerMan::timer("Benchmark/solve/gettriplets").start();
	level0->getTriplets(triplets);
	CSim::TimerMan::timer("Benchmark/solve/gettriplets").stop();

	CSim::TimerMan::timer("Benchmark/solve/build_explicit_matrix").start();
	//Batty matrix
	SparseMatrix<float> lhs;
	lhs.resize(level0->mNumDof);
	for (auto& triplet : triplets) {
		lhs.add_to_element(triplet.row(), triplet.col(), triplet.value());
	}
	CSim::TimerMan::timer("Benchmark/solve/build_explicit_matrix").stop();

	PCGSolver<float> batty_solver;
	batty_solver.set_solver_parameters(1e-7, 2000, 0.97, 0.25);
	float residual;
	int iterations;
	printf("start solving Batty ICPCG... \n");
	CSim::TimerMan::timer("Benchmark/solve/ICPCG").start();
	batty_solver.solve(lhs, vector_rhs, vector_x, residual, iterations);
	CSim::TimerMan::timer("Benchmark/solve/ICPCG").stop();
	std::cout << "ICPCG solve iteration: " << iterations << " residual: " << residual << std::endl;
	CSim::TimerMan::timer("Benchmark/solve/cvtgrid").start();
	level0->vectorToGrid(initial_guess, vector_x);
	CSim::TimerMan::timer("Benchmark/solve/cvtgrid").stop();
}

void solve_poisson_SIMD_UAAMG(
	openvdb::FloatGrid::Ptr rhs,
	openvdb::FloatGrid::Ptr initial_guess,
	simd_uaamg::LaplacianWithLevel::Ptr level0) {

	CSim::TimerMan::timer("Benchmark/solve/build_levels").start();
	simd_uaamg::PoissonSolver simd_solver(level0);
	CSim::TimerMan::timer("Benchmark/solve/build_levels").stop();

	int smoother = YamlSingleton::get()["pressure_smoother"].as<int>();
	switch (smoother) {
	case 0:
		//damped jacobi
		simd_solver.mSmoother = simd_uaamg::PoissonSolver::SmootherOption::WeightedJacobi;
		break;
	case 1:
		simd_solver.mSmoother = simd_uaamg::PoissonSolver::SmootherOption::ScheduledRelaxedJacobi;
		break;
	case 2:
		simd_solver.mSmoother = simd_uaamg::PoissonSolver::SmootherOption::RedBlackGaussSeidel;
		break;
	case 3:
		simd_solver.mSmoother = simd_uaamg::PoissonSolver::SmootherOption::SPAI0;
		break;
	}
	CSim::TimerMan::timer("Benchmark/solve/simd_solve").start();
	initial_guess = level0->getZeroVectorGrid();
	int state = simd_solver.solveMultigridPCG(initial_guess, rhs);
	if (state == simd_uaamg::PoissonSolver::SUCCESS) {
		printf("solve success\n");
	}
	else {
		printf("solve failed\n");
	}
	CSim::TimerMan::timer("Benchmark/solve/simd_solve").stop();
}


void solve_poisson_AMGCL(
	openvdb::FloatGrid::Ptr in_rhs,
	openvdb::FloatGrid::Ptr initial_guess,
	simd_uaamg::LaplacianWithLevel::Ptr level0) {

	std::vector<float> vector_rhs, vector_x;

	CSim::TimerMan::timer("Benchmark/solve/cvtvector").start();
	level0->gridToVector(vector_rhs, in_rhs);
	level0->gridToVector(vector_x, initial_guess);
	CSim::TimerMan::timer("Benchmark/solve/cvtvector").stop();
	//left hand side matrix
	std::vector<Eigen::Triplet<float>> triplets;
	CSim::TimerMan::timer("Benchmark/solve/gettriplets").start();
	level0->getTriplets(triplets);
	CSim::TimerMan::timer("Benchmark/solve/gettriplets").stop();

	CSim::TimerMan::timer("Benchmark/solve/finest_matrix").start();
	//AMGCL matrix
	int ndof = level0->mNumDof;
	std::vector<ptrdiff_t> ptr, col;
	std::vector<float> val;
	ptr.clear(); ptr.reserve(ndof + 1); ptr.push_back(0);
	int prev_row = 0;
	for (auto& triplet : triplets) {
		int row = triplet.row();
		
		if (row != prev_row) {
			//printf("ptr: %d\n", col.size());
			ptr.push_back(col.size());
			prev_row = row;
		}
		//printf("row: %d, col: %d, val: %e\n", triplet.row(), triplet.col(), triplet.value());
		col.push_back(triplet.col());
		val.push_back(triplet.value());
	}
	ptr.push_back(col.size());
	//printf("ptr: %d\n", col.size());

	auto A = std::tie(ndof, ptr, col, val);
	CSim::TimerMan::timer("Benchmark/solve/finest_matrix").stop();

	CSim::TimerMan::timer("Benchmark/solve/build_levels").start();

	// Compose the solver type
	//   the solver backend:
	typedef amgcl::backend::builtin<float> SBackend;
	//   the preconditioner backend:
	typedef amgcl::backend::builtin<float> PBackend;

	int smoother = YamlSingleton::get()["pressure_smoother"].as<int>();

	if (smoother == 0) {
		typedef amgcl::make_solver<
			amgcl::amg<
			PBackend,
			amgcl::coarsening::smoothed_aggregation,
			amgcl::relaxation::damped_jacobi
			>,
			amgcl::solver::cg<SBackend>
		> Solver_damped_jacobi;

		// Initialize the solver with the system matrix:
		Solver_damped_jacobi::params prm;
		prm.solver.tol = 1e-7;
		prm.precond.relax.damping = 6.0 / 7.0;
		Solver_damped_jacobi solve(A, prm);
		CSim::TimerMan::timer("Benchmark/solve/build_levels").stop();

		// Show the mini-report on the constructed solver:
		std::cout << solve << std::endl;

		// Solve the system with the zero initial approximation:
		size_t iters;
		float error;

		CSim::TimerMan::timer("Benchmark/solve/solve").start();
		std::tie(iters, error) = solve(A, vector_rhs, vector_x);
		std::cout << "iters: " << iters << "\terror: " << error << '\n';
		CSim::TimerMan::timer("Benchmark/solve/solve").stop();
	}
	else if (smoother == 3) {
		typedef amgcl::make_solver<
			amgcl::amg<
			PBackend,
			amgcl::coarsening::smoothed_aggregation,
			amgcl::relaxation::spai0
			>,
			amgcl::solver::cg<SBackend>
		> Solver_spai0;

		// Initialize the solver with the system matrix:
		Solver_spai0::params prm;
		prm.solver.tol = 1e-7;
		Solver_spai0 solve(A, prm);
		CSim::TimerMan::timer("Benchmark/solve/build_levels").stop();

		// Show the mini-report on the constructed solver:
		std::cout << solve << std::endl;

		// Solve the system with the zero initial approximation:
		size_t iters;
		float error;

		CSim::TimerMan::timer("Benchmark/solve/solve").start();
		std::tie(iters, error) = solve(A, vector_rhs, vector_x);
		std::cout << "iters: " << iters << "\terror: " << error << '\n';
		CSim::TimerMan::timer("Benchmark/solve/solve").stop();
	}
	else {
		printf("selected smoothing method not implemented\n");
		exit(0);
	}
}


namespace {
struct restriction_triplet_reducer {
	restriction_triplet_reducer(
		const std::vector<openvdb::Int32Tree::LeafNodeType*>& in_child_nodes,
		openvdb::Int32Grid::Ptr in_parent_dof_idx):
	m_child_nodes(in_child_nodes),
	m_parent_dof_idx(in_parent_dof_idx){

	}
	restriction_triplet_reducer(const restriction_triplet_reducer& other, tbb::split):
	m_child_nodes(other.m_child_nodes),
	m_parent_dof_idx(other.m_parent_dof_idx){
	}

	void operator() (const tbb::blocked_range<size_t>& range){
		auto parent_dof_axr{ m_parent_dof_idx->getConstUnsafeAccessor() };
		for (auto i = range.begin(); i != range.end(); ++i) {
			for (auto iter = m_child_nodes[i]->beginValueOn(); iter; ++iter) {
				auto base_coord = iter.getCoord() + iter.getCoord();
				for (int ii = 0; ii < 2; ii++) {
					for (int jj = 0; jj < 2; jj++) {
						for (int kk = 0; kk < 2; kk++) {
							int parent_dof_idx = parent_dof_axr.getValue(base_coord.offsetBy(ii, jj, kk));
							if (-1 != parent_dof_idx) {
								m_triplets.push_back({ iter.getValue(),parent_dof_idx, 0.125f });
							}
						}//end kk
					}//end jj
				}//end ii
			}//for all active voxel
		}//for each leaf
	}//operator()

	void join(const restriction_triplet_reducer& other) {
		size_t original_size = m_triplets.size();
		size_t extra_size = other.m_triplets.size();
		m_triplets.resize(original_size + extra_size);
		std::copy(other.m_triplets.begin(), other.m_triplets.end(), m_triplets.begin() + original_size);
	}
	openvdb::Int32Grid::Ptr m_parent_dof_idx;
	const std::vector<openvdb::Int32Tree::LeafNodeType*>& m_child_nodes;
	std::vector<Eigen::Triplet<float>> m_triplets;
};

}


void Eigen_Jacobi_apply(Eigen::VectorXf& out_result, Eigen::VectorXf& in_lhs, Eigen::VectorXf& in_rhs, const Eigen::SparseMatrix<float, Eigen::RowMajor>& LHS, float omega) {
	ZoneScoped
	int ndof = LHS.rows();

	tbb::parallel_for(0, ndof, [&](int i) {
		float diagonal=1.0f;
		float result0=0.f;
		for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(LHS, i); it; ++it) {
			//it.value();
			//it.row();   // row index
			//it.col();   // col index (here it is equal to k)
			//it.index(); // inner index, here it is equal to it.row()

			result0 += in_lhs[it.col()] * it.value();
			if (it.col() == i) {
				diagonal = it.value();
			}
		}

		float residual = in_rhs[i] - result0;
		if (diagonal != 0.f) {
			residual = residual / diagonal;
			out_result[i] = in_lhs[i] + residual * omega;
		}
		else {
			out_result[i] = in_lhs[i];
		}
		});
}

void Eigen_SPAI0_apply(Eigen::VectorXf& out_result, Eigen::VectorXf& in_lhs, Eigen::VectorXf& in_rhs, const Eigen::SparseMatrix<float, Eigen::RowMajor>& LHS) {
	ZoneScoped
	int ndof = LHS.rows();

	//Eigen::VectorXf spai0_m = Eigen::VectorXf::Zero(ndof);

	tbb::parallel_for(0, ndof, [&](int i) {
		float sum_row_square = 0.f;
		float diagonal = 1.0f;
		float result0 = 0.f;
		for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(LHS, i); it; ++it) {
			//it.value();
			//it.row();   // row index
			//it.col();   // col index (here it is equal to k)
			//it.index(); // inner index, here it is equal to it.row()
			sum_row_square += (it.value()* it.value());
			result0 += in_lhs[it.col()] * it.value();
			if (it.col() == i) {
				diagonal = it.value();
			}
		}

		float residual = in_rhs[i] - result0;
		if (diagonal != 0.f) {
			float spai0_m_i = diagonal / sum_row_square;
			out_result[i] = in_lhs[i] +  residual* spai0_m_i;
		}
		else {
			out_result[i] = in_lhs[i];
		}
		});
}

void Eigen_residual_apply(Eigen::VectorXf& out_result, Eigen::VectorXf& in_lhs, Eigen::VectorXf& in_rhs, const Eigen::SparseMatrix<float, Eigen::RowMajor>& LHS) {
	ZoneScoped
		int ndof = LHS.rows();

	tbb::parallel_for(0, ndof, [&](int i) {
		float diagonal = 1.0f;
		float result0 = 0.f;
		for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(LHS, i); it; ++it) {
			//it.value();
			//it.row();   // row index
			//it.col();   // col index (here it is equal to k)
			//it.index(); // inner index, here it is equal to it.row()

			result0 += in_lhs[it.col()] * it.value();
		}

		out_result[i] = in_rhs[i] - result0;
		});
}

template <bool forward>
void Eigen_RBGS_apply(Eigen::VectorXf& in_out_lhs, Eigen::VectorXf& in_rhs, const std::vector<int>& color_update_mask, const Eigen::SparseMatrix<float, Eigen::RowMajor>& LHS, float omega, int ncolor) {
	ZoneScoped;
	int ndof = LHS.rows();

	//red update pass
	for (int icolor = 0; icolor < ncolor; ++icolor) {
		int actual_color = icolor;
		if (!forward) {
			actual_color = ncolor - 1 - icolor;
		}
		tbb::parallel_for(0, ndof, [&](int i) {
			if (actual_color != color_update_mask[i]) {
				return;
			}

			//float diagonal = 1.0f;
			//float result0 = 0.f;
			//for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(LHS, i); it; ++it) {
			//	result0 += in_out_lhs[it.col()] * it.value();
			//	
			//	if (it.col() == i) {
			//		diagonal = it.value();
			//	}
			//}

			//float residual = in_rhs[i] - result0;
			//if (diagonal != 0.f) {
			//	residual = residual / diagonal;
			//	in_out_lhs[i] = in_out_lhs[i] + residual * omega;
			//}

			float diagonal = 0.0f;
			float off_diag_result = 0.f;
			for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(LHS, i); it; ++it) {

				if (it.col() == i) {
					diagonal = it.value();
				}
				else {
					off_diag_result += in_out_lhs[it.col()] * it.value();
				}
			}

			float residual = in_rhs[i] - off_diag_result;
			if (diagonal != 0.f) {
				residual = omega * residual / diagonal;
				in_out_lhs[i] = (1.0f - omega) * in_out_lhs[i] + residual;
			}

		});
	}
}

namespace {
struct explicit_PCG_solver {
	enum class Smoother {
		SCHEDULED_RELAXED_JACOBI,
		DAMPED_JACOBI,
		RED_BLACK_GAUSS_SEIDEL,
		SPAI0
	}; 

	using vecEigenMat = std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor>>;
	explicit_PCG_solver(const vecEigenMat& LHS_matrix, 
		const vecEigenMat& restriction_matrix,
		const vecEigenMat& prolongation_matrix, 
		const std::vector<std::vector<int>>& color_update_mask,
		int ncolor_) : 
			m_LHS_matrix(LHS_matrix),
			m_restriction_matrix(restriction_matrix),
			m_prolongation_matrix(prolongation_matrix), 
			m_color_update_mask(color_update_mask),
			ncolor(ncolor_)
	{
		m_iteration = 0;
		m_max_iter = 100;
		m_tolerance = 1e-7f;
		m_smoother = Smoother::SCHEDULED_RELAXED_JACOBI;
		m_sor_coef = 1.2;
		m_smooth_count = 3;
		m_damped_jacobi_coef = 6.0 / 7.0;
		mu = 2;
		for (int level = 0; level < m_LHS_matrix.size(); level++) {
			//the solution at each level
			m_mucycle_lhss.push_back(Eigen::VectorXf::Zero(m_LHS_matrix[level].rows()));
			//the right hand side at each level
			m_mucycle_rhss.push_back(m_mucycle_lhss.back());
			//the temporary result to store the jacobi iteration
			//use std::shared_ptr::swap to change the content
			m_mucycle_temps.push_back(m_mucycle_lhss.back());
		}
		// constructCoarsestLevelExactSolver
		m_coarsest_CG_solver = std::make_shared<Eigen::ConjugateGradient<Eigen::SparseMatrix<float>>>(m_LHS_matrix.back());
		m_coarsest_CG_solver->setMaxIterations(10);
	}

	void mucycle_preconditioner(Eigen::VectorXf& in_out_lhs, const Eigen::VectorXf& in_rhs, const int level, const int n, const int mu_time) {
		ZoneScoped;
		auto get_scheduled_weight = [n](int iteration) {
			std::array<float, 3> scheduled_weight;
			if (iteration >= n) {
				return 6.0f / 7.0f;
			}
			if (n == 1) {
				scheduled_weight[0] = 6.0f / 7.0f;
			}
			if (n == 2) {
				scheduled_weight[0] = 1.7319f;
				scheduled_weight[1] = 0.5695f;
			}
			if (n == 3) {
				scheduled_weight[0] = 2.2473f;
				scheduled_weight[1] = 0.8571f;
				scheduled_weight[2] = 0.5296f;
			}
			return scheduled_weight[iteration];
		};

		size_t nlevel = m_LHS_matrix.size();
		//in_out_lhs = in_rhs;
		//return;
		if (level == nlevel - 1) {
			in_out_lhs = m_coarsest_CG_solver->solve(in_rhs);
			return;
		}
		
		m_mucycle_lhss[level] = in_out_lhs;
		m_mucycle_rhss[level] = in_rhs;

		for (int i = 0; i < n; i++) {
			switch (m_smoother) {
			default:
			case Smoother::SCHEDULED_RELAXED_JACOBI:
				Eigen_Jacobi_apply(m_mucycle_temps[level], m_mucycle_lhss[level], m_mucycle_rhss[level], m_LHS_matrix[level], get_scheduled_weight(i));
				m_mucycle_temps[level].swap(m_mucycle_lhss[level]);
				break;
			case Smoother::DAMPED_JACOBI:
				Eigen_Jacobi_apply(m_mucycle_temps[level], m_mucycle_lhss[level], m_mucycle_rhss[level], m_LHS_matrix[level], m_damped_jacobi_coef);
				m_mucycle_temps[level].swap(m_mucycle_lhss[level]);
				break;
			case Smoother::RED_BLACK_GAUSS_SEIDEL:
				Eigen_RBGS_apply<true>(m_mucycle_lhss[level], m_mucycle_rhss[level], m_color_update_mask[level], m_LHS_matrix[level], m_sor_coef, ncolor);
				break;
			case Smoother::SPAI0:
				Eigen_SPAI0_apply(m_mucycle_temps[level], m_mucycle_lhss[level], m_mucycle_rhss[level], m_LHS_matrix[level]);
				m_mucycle_temps[level].swap(m_mucycle_lhss[level]);
				break;
			}
			//printf("level%d forward iter: %d residual:%e\n",level, i, (mMuCycleRHSs[level] - m_LHS_matrix[level] * mMuCycleLHSs[level]).norm());
		}
		
		//mMuCycleTemps[level] = mMuCycleRHSs[level] - m_LHS_matrix[level] * mMuCycleLHSs[level];
		Eigen_residual_apply(m_mucycle_temps[level], m_mucycle_lhss[level], m_mucycle_rhss[level], m_LHS_matrix[level]);
		
		

		int child_level = level + 1;
		{
			ZoneNamedN(restriction, "restriction", true);
			m_mucycle_rhss[child_level] = m_restriction_matrix[level] * m_mucycle_temps[level];
		}
		

		m_mucycle_lhss[child_level].setZero();
		mucycle_preconditioner(m_mucycle_lhss[child_level], m_mucycle_rhss[child_level], child_level, n, mu_time);
		for (int mu = 1; mu < mu_time; mu++) {
			mucycle_preconditioner(m_mucycle_lhss[child_level], m_mucycle_rhss[child_level], child_level, n, mu_time);
		}

		{
			ZoneNamedN(prolongation, "prolongation", true);
			m_mucycle_lhss[level] += m_prolongation_matrix[level] * m_mucycle_lhss[child_level];
		}
		

		for (int i = 0; i < n; i++) {
			switch (m_smoother) {
			default:
			case Smoother::SCHEDULED_RELAXED_JACOBI:
				Eigen_Jacobi_apply(m_mucycle_temps[level], m_mucycle_lhss[level], m_mucycle_rhss[level], m_LHS_matrix[level], get_scheduled_weight(n - 1 - i));
				m_mucycle_temps[level].swap(m_mucycle_lhss[level]);
				break;
			case Smoother::DAMPED_JACOBI:
				Eigen_Jacobi_apply(m_mucycle_temps[level], m_mucycle_lhss[level], m_mucycle_rhss[level], m_LHS_matrix[level], m_damped_jacobi_coef);
				m_mucycle_temps[level].swap(m_mucycle_lhss[level]);
				break;
			case Smoother::RED_BLACK_GAUSS_SEIDEL:
				Eigen_RBGS_apply<false>(m_mucycle_lhss[level], m_mucycle_rhss[level], m_color_update_mask[level], m_LHS_matrix[level], m_sor_coef, ncolor);
				break;
			case Smoother::SPAI0:
				Eigen_SPAI0_apply(m_mucycle_temps[level], m_mucycle_lhss[level], m_mucycle_rhss[level], m_LHS_matrix[level]);
				m_mucycle_temps[level].swap(m_mucycle_lhss[level]);
				break;
			}
			//printf("level%d back iter: %d residual:%e\n", level, i, (mMuCycleRHSs[level] - m_LHS_matrix[level] * mMuCycleLHSs[level]).norm());
		}

		in_out_lhs = m_mucycle_lhss[level];
	}

	float abs_max(const Eigen::VectorXf& x)
	{
		ZoneScoped;
		size_t n = x.size();
		//return cblas_idamax((int)x.size(), &x[0], 1); 
		return tbb::parallel_reduce(
			tbb::blocked_range<size_t>(0, n),
			0.f,
			[&](const tbb::blocked_range<size_t>& r, float max_value) {
				for (size_t a = r.begin(); a != r.end(); ++a) {
					if (std::abs(x[a]) > std::abs(max_value)) {
						max_value = std::abs(x[a]);
					}
				}
				return max_value;
			},
			[](float max_value0, float max_value1)->float {
				if (max_value0 > max_value1) {
					return max_value0;
				}
				else {
					return max_value1;
				}
			}
			);
	}

	float dot(const Eigen::VectorXf& x, const Eigen::VectorXf& y)
	{
		ZoneScoped;
		size_t n = x.size();
		//return cblas_ddot((int)x.size(), &x[0], 1, &y[0], 1); 
		return tbb::parallel_reduce(
			tbb::blocked_range<size_t>(0, n),
			0.f,
			[&](const tbb::blocked_range<size_t>& r, float init) {
				for (size_t a = r.begin(); a != r.end(); ++a) {
					init += x[a] * y[a];
				}
				return init;
			},
			[](float x, float y)->float {
				return x + y;
			}
			);
		/*double sum = 0;
		for(int i = 0; i < x.size(); ++i)
		   sum += x[i]*y[i];
		return sum;*/
	}

	int pcg_solve(Eigen::VectorXf& in_out_presssure, Eigen::VectorXf& in_rhs)
	{
		ZoneScoped;
		m_iteration = 0;

		//according to mcadams algorithm 3
		//line2
		Eigen::VectorXf r = Eigen::VectorXf::Zero(m_LHS_matrix[0].rows());
		r =  in_rhs - m_LHS_matrix[0]* in_out_presssure;
		float nu = abs_max(r);
		float numax = m_tolerance * nu; //numax = std::min(numax, 1e-7f);

		//line3
		if (nu <= numax) {
			printf("initial iter:%d err:%e\n", m_iteration, nu);
			printf("r norm:%e\n", r.norm());
			return 0;
		}

		//line4
		Eigen::VectorXf p = Eigen::VectorXf::Zero(m_LHS_matrix[0].rows());
		mucycle_preconditioner(p, r, 0, m_smooth_count, mu);
		float rho = dot(p, r);

		Eigen::VectorXf z = Eigen::VectorXf::Zero(m_LHS_matrix[0].rows());
		//line 5
		for (; m_iteration < m_max_iter; m_iteration++) {
			//line6
			{
				ZoneNamedN(matrixMul,"matrixMul", true);
				z = m_LHS_matrix[0] * p;
			}
			float sigma = dot(p, z);
			//line7
			float alpha = rho / sigma;
			//printf("alpha%e \n", alpha);
			//line8
			{
				ZoneNamedN(updateR, "updateR", true);
				r -= alpha * z;
			}
			
			nu = abs_max(r); printf("iter:%d err:%e\n", m_iteration + 1, nu);
			//line9
			if (nu <= numax || m_iteration == (m_max_iter - 1)) {
				//line10
				in_out_presssure += alpha*p;
				//line11
				printf("final iter:%d err:%e\n", m_iteration + 1, nu);
				return 0;
				//line12
			}
			//line13
			z.setZero();
			mucycle_preconditioner(z, r, 0, m_smooth_count, mu);

			float rho_new = dot(z, r);

			//line14
			float beta = rho_new / rho;
			//line15
			rho = rho_new;
			//line16
			{
				ZoneNamedN(update, "update", true);
				in_out_presssure += alpha * p;
				p = z + beta * p;
			}
			
			//line17
		}

		//line18
		return 1;
	}

	std::vector<Eigen::VectorXf> m_mucycle_lhss, m_mucycle_temps, m_mucycle_rhss;
	const std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor>> &m_LHS_matrix, &m_restriction_matrix, &m_prolongation_matrix;
	const std::vector<std::vector<int>>& m_color_update_mask;

	Eigen::VectorXf m_coarsest_eigen_rhs, m_coarsest_eigen_solution;
	std::shared_ptr<Eigen::ConjugateGradient<Eigen::SparseMatrix<float>>> m_coarsest_CG_solver;
	int ncolor;

	uint32_t m_iteration;
	uint32_t m_max_iter;
	float m_tolerance;
	Smoother m_smoother;
	float m_sor_coef;
	float m_damped_jacobi_coef;
	int m_smooth_count;
	int mu;
};
}



void solve_poisson_UAAMG(openvdb::FloatGrid::Ptr rhs,
	openvdb::FloatGrid::Ptr initial_guess,
	simd_uaamg::LaplacianWithLevel::Ptr level0) {
	
	//construct levels
	CSim::TimerMan::timer("Benchmark/solve/SIMD_UAAMGinit").start();
	simd_uaamg::PoissonSolver simd_solver(level0);
	CSim::TimerMan::timer("Benchmark/solve/SIMD_UAAMGinit").stop();


	//convert each level to a sparse matrix
	std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor>> LHS_matrix, restriction_matrix, prolongation_matrix;
	std::vector<std::vector<int>> red_update_mask;

	CSim::TimerMan::timer("Benchmark/solve/finest_matrix").start();
	std::vector<Eigen::Triplet<float>> lv0_LHS_triplet;
	CSim::TimerMan::timer("Benchmark/solve/finest_matrix/get_triplet").start();
	simd_solver.mMultigridHierarchy[0]->getTriplets(lv0_LHS_triplet);
	CSim::TimerMan::timer("Benchmark/solve/finest_matrix/get_triplet").stop();

	CSim::TimerMan::timer("Benchmark/solve/finest_matrix/build_matrix").start();
	int lv0_dof = simd_solver.mMultigridHierarchy[0]->mNumDof;
	LHS_matrix.push_back(Eigen::SparseMatrix<float, Eigen::RowMajor>{});
	LHS_matrix.back().resize(lv0_dof, lv0_dof);
	LHS_matrix.back().setFromTriplets(lv0_LHS_triplet.begin(), lv0_LHS_triplet.end());
	CSim::TimerMan::timer("Benchmark/solve/finest_matrix/build_matrix").stop();
	CSim::TimerMan::timer("Benchmark/solve/finest_matrix").stop();

	CSim::TimerMan::timer("Benchmark/solve/build_levels").start();
	for (int i = 0; i < simd_solver.mMultigridHierarchy.size() - 1; i++) {
		CSim::TimerMan::timer("Benchmark/solve/build_levels/lv" + std::to_string(i)).start();
		//generate the triplets for this level of matrix
		int this_level_dof = simd_solver.mMultigridHierarchy[i]->mNumDof;

		red_update_mask.push_back(std::vector<int>{});
		red_update_mask.back().resize(this_level_dof);

		auto red_update_mask_setter = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index32) {
			for (auto iter = leaf.beginValueOn(); iter; ++iter) {
				auto gcoord = iter.getCoord();
				auto sum = gcoord.x() + gcoord.y() + gcoord.z();
				if (sum % 2 == 0) {
					red_update_mask.back()[iter.getValue()] = 0;
				}
				else {
					red_update_mask.back()[iter.getValue()] = 1;
				}
			}
		};
		CSim::TimerMan::timer("Benchmark/solve/build_levels/lv" + std::to_string(i) + "/rbmask").start();
		simd_solver.mMultigridHierarchy[i]->mDofLeafManager->foreach(red_update_mask_setter);
		CSim::TimerMan::timer("Benchmark/solve/build_levels/lv" + std::to_string(i) + "/rbmask").stop();

		//generate triplets for restriction and prolongation matrix
		int child_level = i + 1;
		if (child_level < simd_solver.mMultigridHierarchy.size()) {
			auto child_laplacian = simd_solver.mMultigridHierarchy[child_level];
			auto child_dofidx = child_laplacian->mDofIndex;
			//get all child nodes
			std::vector<openvdb::Int32Tree::LeafNodeType*> child_nodes;
			child_nodes.reserve(child_dofidx->tree().leafCount());
			child_dofidx->tree().getNodes(child_nodes);

			size_t nchild_nodes = child_nodes.size();

			restriction_triplet_reducer triplet_reducer(child_nodes, simd_solver.mMultigridHierarchy[i]->mDofIndex);
			CSim::TimerMan::timer("Benchmark/solve/build_levels/lv" + std::to_string(i) + "/Rtriplet").start();
			tbb::parallel_reduce(tbb::blocked_range<size_t>(0, nchild_nodes), triplet_reducer);
			CSim::TimerMan::timer("Benchmark/solve/build_levels/lv" + std::to_string(i) + "/Rtriplet").stop();
			CSim::TimerMan::timer("Benchmark/solve/build_levels/lv" + std::to_string(i) + "/Rbuild").start();
			restriction_matrix.push_back(Eigen::SparseMatrix<float, Eigen::RowMajor>{});
			restriction_matrix.back().resize(child_laplacian->mNumDof, this_level_dof);
			restriction_matrix.back().setFromTriplets(triplet_reducer.m_triplets.begin(), triplet_reducer.m_triplets.end());
			CSim::TimerMan::timer("Benchmark/solve/build_levels/lv" + std::to_string(i) + "/Rbuild").stop();
			CSim::TimerMan::timer("Benchmark/solve/build_levels/lv" + std::to_string(i) + "/Pbuild").start();
			prolongation_matrix.push_back(restriction_matrix.back().transpose() * 8.0f);
			CSim::TimerMan::timer("Benchmark/solve/build_levels/lv" + std::to_string(i) + "/Pbuild").stop();
		}
		CSim::TimerMan::timer("Benchmark/solve/build_levels/lv" + std::to_string(i) + "/RAP").start();
		LHS_matrix.push_back(restriction_matrix.back()*LHS_matrix.back()*prolongation_matrix.back()*0.5);
		CSim::TimerMan::timer("Benchmark/solve/build_levels/lv" + std::to_string(i) + "/RAP").stop();

		CSim::TimerMan::timer("Benchmark/solve/build_levels/lv" + std::to_string(i)).stop();
	}
	CSim::TimerMan::timer("Benchmark/solve/build_levels").stop();
	std::vector<float> vector_rhs, vector_x;

	
	level0->gridToVector(vector_rhs, rhs);
	level0->gridToVector(vector_x, initial_guess);
	

	//right hand side, initial guess
	Eigen::VectorXf eigen_rhs, eigen_x;
	eigen_rhs = Eigen::Map<Eigen::VectorXf>(vector_rhs.data(), vector_rhs.size());
	eigen_x = Eigen::Map<Eigen::VectorXf>(vector_x.data(), vector_x.size());

	explicit_PCG_solver UAAMGPCG(LHS_matrix, restriction_matrix, prolongation_matrix, red_update_mask, 2);
	int smoother = YamlSingleton::get()["pressure_smoother"].as<int>();
	switch (smoother) {
	case 0:
		//damped jacobi
		UAAMGPCG.m_smoother = explicit_PCG_solver::Smoother::DAMPED_JACOBI;
		break;
	case 1:
		UAAMGPCG.m_smoother = explicit_PCG_solver::Smoother::SCHEDULED_RELAXED_JACOBI;
		break;
	case 2:
		UAAMGPCG.m_smoother = explicit_PCG_solver::Smoother::RED_BLACK_GAUSS_SEIDEL;
		break;
	case 3:
		UAAMGPCG.m_smoother = explicit_PCG_solver::Smoother::SPAI0;
		break;
	}
	CSim::TimerMan::timer("Benchmark/solve/solve").start();
	UAAMGPCG.pcg_solve(eigen_x, eigen_rhs);
	CSim::TimerMan::timer("Benchmark/solve/solve").stop();
}

void benchmark_poisson() {
	//prepare the liquid sdf and solid sdf

	openvdb::Vec3f cube_size{ 1.0f };
	printf("parse nx:\n");
	for (YAML::const_iterator it = YamlSingleton::get().begin(); it != YamlSingleton::get().end(); ++it) {
		std::cout << "item " << it->first.as<std::string>() << " is " << it->second << "\n";
	}
	int nx = YamlSingleton::get()["nx"].as<int>();
	printf("nx:%d\n", nx);
	float dx = 1.0f / nx;
	CSim::TimerMan::timer("Benchmark/SDF_setup").start();
	auto liquid_sdf = openvdb::createLevelSet<openvdb::FloatGrid>(dx);
	liquid_sdf->setName("liquid_sdf");
	auto solid_sdf = openvdb::createLevelSet<openvdb::FloatGrid>(dx);
	solid_sdf->setName("solid_sdf");

	auto poisson_type = YamlSingleton::get()["poisson_scene_type"].as<std::string>();
	std::string output_prefix = YamlSingleton::get()["output_prefix"].as<std::string>();
	std::filesystem::path poisson_output_directory(output_prefix);
	std::filesystem::create_directory(poisson_output_directory);

	if (poisson_type == "compact") {
		compact_poisson_scene(liquid_sdf, solid_sdf, dx, cube_size); 
		poisson_output_directory /= ("pressure_" + poisson_type + std::to_string(nx));
	}
	if (poisson_type == "sparse") {
		sparse_poisson_scene(liquid_sdf, solid_sdf, dx, cube_size);
		int nballs = YamlSingleton::get()["nballs"].as<int>();
		poisson_output_directory /= ("pressure_" + poisson_type + std::to_string(nx) + "_nballs" + std::to_string(nballs));
	}
	CSim::TimerMan::timer("Benchmark/SDF_setup").stop();
	std::filesystem::create_directory(poisson_output_directory);
	std::filesystem::path output_vdb = poisson_output_directory / "sdf.vdb";
	std::cout << "SDF_setup time: " << CSim::TimerMan::timer("Benchmark/SDF_setup").lasttime() << " seconds" << std::endl;

	//solve the equation with different method.
	float dt = 1.0f;
	CSim::TimerMan::timer("Benchmark/faceweight").start();
	auto face_weight = FLIP_Solver_OpenVDB::calculate_face_weights(liquid_sdf, solid_sdf);
	CSim::TimerMan::timer("Benchmark/faceweight").stop();
	std::cout << "faceweight time: " << CSim::TimerMan::timer("Benchmark/faceweight").lasttime() << " seconds" << std::endl;

	//build the system matrix
	CSim::TimerMan::timer("Benchmark/build_matrix").start();
	auto level0 = simd_uaamg::LaplacianWithLevel::createPressurePoissonLaplacian(liquid_sdf, face_weight, dt);
	printf("ndof:%d\n", level0->mNumDof);
	CSim::TimerMan::timer("Benchmark/build_matrix").stop();
	std::cout << "build matrix time: " << CSim::TimerMan::timer("Benchmark/build_matrix").lasttime() << " seconds" << std::endl;

	float active_portion = sparsity_fraction_dof(level0->mDofIndex->tree());
	level0->mDofIndex->setName("dofidx0");
	printf("active portion:%f\n", active_portion);

	CSim::TimerMan::timer("Benchmark/random_rhs").start();
	//build the rhs
	auto rhs = level0->getZeroVectorGrid();
	rhs->setName("rhs");
	//set a random initial guess
	auto pressure = rhs->deepCopy();
	pressure->setName("pressure");
	int ndof = level0->mNumDof;
	auto leaf_randomizer = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) {
		auto result_leaf = rhs->tree().probeLeaf(leaf.origin());
		std::mt19937 generator(/*seed=*/leafpos);
		std::uniform_real_distribution<> distribution(-1, 0);
		for (auto iter = result_leaf->beginValueOn(); iter; ++iter) {
			iter.setValue(distribution(generator) + leaf.getValue(iter.offset()) / (float)ndof);
		}
	};
	level0->mDofLeafManager->foreach(leaf_randomizer);
	CSim::TimerMan::timer("Benchmark/random_rhs").stop();

	CSim::TimerMan::timer("Benchmark/solve").start();
	//dispatch to different kind of solver
	int solver_method = YamlSingleton::get()["pressure_solver_method"].as<int>();
	
	switch (solver_method) {
	case 0:
		//Eigen Jacobi CG
		solve_poisson_Eigen_diagonalPCG(rhs, pressure, level0);
		break;
	case 1:
		//Eigen ICPCG
		solve_poisson_Eigen_ICPCG(rhs, pressure, level0);
		break;
	case 2:
		//Batty ICPCG
		solve_poisson_Batty_ICPCG(rhs, pressure, level0);
		break;
	case 3:
		//UAAMG
		solve_poisson_UAAMG(rhs, pressure, level0);
		break;
	case 4:
	default:
		//SIMD-UAAMG
		solve_poisson_SIMD_UAAMG(rhs, pressure, level0);
		break;
	case 5:
		//AMGCL
		solve_poisson_AMGCL(rhs, pressure, level0);
		break;
	}
	
	printf("solve done\n");
	CSim::TimerMan::timer("Benchmark/solve").stop();
	CSim::TimerMan::timer("Benchmark/IO").start();
	openvdb::io::File(output_vdb.string()).write({ liquid_sdf, solid_sdf, pressure });
	CSim::TimerMan::timer("Benchmark/IO").stop();
}

void solve_viscosity_Eigen_diagonalPCG(packed_FloatGrid3 rhs,
	packed_FloatGrid3 initial_guess,
	simd_uaamg::L_with_level* level0) {

	Eigen::VectorXf eigen_rhs, eigen_x;

	CSim::TimerMan::timer("Benchmark/solve/cvtvector").start();
	level0->to_Eigenvector(eigen_rhs, rhs);
	level0->to_Eigenvector(eigen_x, initial_guess);
	CSim::TimerMan::timer("Benchmark/solve/cvtvector").stop();

	//left hand side matrix
	std::vector<Eigen::Triplet<float>> triplets;
	CSim::TimerMan::timer("Benchmark/solve/gettriplets").start();
	level0->get_triplets(triplets);
	CSim::TimerMan::timer("Benchmark/solve/gettriplets").stop();
	CSim::TimerMan::timer("Benchmark/solve/build_explicit_matrix").start();
	Eigen::SparseMatrix<float, Eigen::RowMajor> eigen_lhs;
	eigen_lhs.resize(eigen_rhs.size(), eigen_rhs.size());
	eigen_lhs.setFromTriplets(triplets.begin(), triplets.end());
	CSim::TimerMan::timer("Benchmark/solve/build_explicit_matrix").stop();

	//the solver
	CSim::TimerMan::timer("Benchmark/solve/CG_init").start();
	Eigen::ConjugateGradient<Eigen::SparseMatrix<float, Eigen::RowMajor>, Eigen::Lower | Eigen::Upper, Eigen::DiagonalPreconditioner<float>> diagonal_PCG(eigen_lhs);
	CSim::TimerMan::timer("Benchmark/solve/CG_init").stop();
	printf("start solving DPCG... with %d threads\n", Eigen::nbThreads());
	std::cout << "Eigen DPCG tolerance: " << diagonal_PCG.tolerance() << std::endl;
	CSim::TimerMan::timer("Benchmark/solve/Eigen_DPCG").start();
	eigen_x = diagonal_PCG.solveWithGuess(eigen_rhs, eigen_x);
	CSim::TimerMan::timer("Benchmark/solve/Eigen_DPCG").stop();
	CSim::TimerMan::timer("Benchmark/solve/cvtgrid").start();
	std::cout << "Eigen DPCG solve iteration: " << diagonal_PCG.iterations() << std::endl;
	std::cout << "Eigen DPCG residual: " << diagonal_PCG.error() << std::endl;
	level0->write_to_FloatGrid3(initial_guess, eigen_x);
	CSim::TimerMan::timer("Benchmark/solve/cvtgrid").stop();
}

void solve_viscosity_Eigen_ICPCG(packed_FloatGrid3 rhs,
	packed_FloatGrid3 initial_guess,
	simd_uaamg::L_with_level* level0) {
}

void solve_viscosity_Batty_ICPCG(packed_FloatGrid3 rhs,
	packed_FloatGrid3 initial_guess,
	simd_uaamg::L_with_level* level0) {
	Eigen::VectorXf eigen_rhs, eigen_x;
	std::vector<float> vector_rhs, vector_x;

	CSim::TimerMan::timer("Benchmark/solve/cvtvector").start();
	level0->to_Eigenvector(eigen_rhs, rhs);
	level0->to_Eigenvector(eigen_x, initial_guess);
	vector_rhs.resize(eigen_rhs.size());
	vector_x.resize(eigen_x.size());
	std::copy(eigen_rhs.data(), eigen_rhs.data() + eigen_rhs.size(), vector_rhs.begin());
	std::copy(eigen_x.data(), eigen_x.data() + eigen_x.size(), vector_x.begin());
	CSim::TimerMan::timer("Benchmark/solve/cvtvector").stop();

	//left hand side matrix
	std::vector<Eigen::Triplet<float>> triplets;
	CSim::TimerMan::timer("Benchmark/solve/gettriplets").start();
	level0->get_triplets(triplets);
	CSim::TimerMan::timer("Benchmark/solve/gettriplets").stop();

	//Batty matrix
	CSim::TimerMan::timer("Benchmark/solve/build_explicit_matrix").start();
	SparseMatrix<float> lhs;
	lhs.resize(level0->m_ndof);
	for (auto& triplet : triplets) {
		lhs.add_to_element(triplet.row(), triplet.col(), triplet.value());
	}
	CSim::TimerMan::timer("Benchmark/solve/build_explicit_matrix").stop();

	PCGSolver<float> batty_solver;
	batty_solver.set_solver_parameters(1e-7, 5000, 0.97, 0.25);
	float residual;
	int iterations;
	printf("start solving Batty ICPCG... \n");
	CSim::TimerMan::timer("Benchmark/solve/ICPCG").start();
	batty_solver.solve(lhs, vector_rhs, vector_x, residual, iterations);
	CSim::TimerMan::timer("Benchmark/solve/ICPCG").stop();

	std::cout << "ICPCG solve iteration: " << iterations << " residual: " << residual << std::endl;
	CSim::TimerMan::timer("Benchmark/solve/cvtgrid").start();
	std::copy(vector_x.begin(), vector_x.end(), eigen_x.data());
	level0->write_to_FloatGrid3(initial_guess, eigen_x);
	CSim::TimerMan::timer("Benchmark/solve/cvtgrid").stop();
}

void solve_viscosity_amgcl(packed_FloatGrid3 in_rhs,
	packed_FloatGrid3 initial_guess,
	simd_uaamg::L_with_level* level0) {
	Eigen::VectorXf eigen_rhs, eigen_x;
	std::vector<float> vector_rhs, vector_x;

	CSim::TimerMan::timer("Benchmark/solve/cvtvector").start();
	level0->to_Eigenvector(eigen_rhs, in_rhs);
	level0->to_Eigenvector(eigen_x, initial_guess);
	vector_rhs.resize(eigen_rhs.size());
	vector_x.resize(eigen_x.size());
	std::copy(eigen_rhs.data(), eigen_rhs.data() + eigen_rhs.size(), vector_rhs.begin());
	std::copy(eigen_x.data(), eigen_x.data() + eigen_x.size(), vector_x.begin());
	CSim::TimerMan::timer("Benchmark/solve/cvtvector").stop();

	//left hand side matrix
	std::vector<Eigen::Triplet<float>> triplets;
	CSim::TimerMan::timer("Benchmark/solve/gettriplets").start();
	level0->get_triplets(triplets);
	CSim::TimerMan::timer("Benchmark/solve/gettriplets").stop();

	CSim::TimerMan::timer("Benchmark/solve/finest_matrix").start();
	//AMGCL matrix
	int ndof = level0->m_ndof;
	std::vector<ptrdiff_t> ptr, col;
	std::vector<float> val;
	ptr.clear(); ptr.reserve(ndof + 1); ptr.push_back(0);
	int prev_row = 0;
	for (auto& triplet : triplets) {
		int row = triplet.row();

		if (row != prev_row) {
			//printf("ptr: %d\n", col.size());
			ptr.push_back(col.size());
			prev_row = row;
		}
		//printf("row: %d, col: %d, val: %e\n", triplet.row(), triplet.col(), triplet.value());
		col.push_back(triplet.col());
		val.push_back(triplet.value());
	}
	ptr.push_back(col.size());
	//printf("ptr: %d\n", col.size());

	auto A = std::tie(ndof, ptr, col, val);
	CSim::TimerMan::timer("Benchmark/solve/finest_matrix").stop();

	CSim::TimerMan::timer("Benchmark/solve/build_levels").start();

	// Compose the solver type
	//   the solver backend:
	typedef amgcl::backend::builtin<float> SBackend;
	//   the preconditioner backend:
	typedef amgcl::backend::builtin<float> PBackend;

	int smoother = YamlSingleton::get()["viscosity_smoother"].as<int>();
	if (smoother == 0) {
		typedef amgcl::make_solver<
			amgcl::amg<
			PBackend,
			amgcl::coarsening::smoothed_aggregation,
			amgcl::relaxation::damped_jacobi
			//amgcl::relaxation::spai0
			>,
			amgcl::solver::cg<SBackend>
		> Solver;

		// Initialize the solver with the system matrix:
		Solver::params prm;
		prm.solver.tol = 1e-7;
		//prm.precond.relax.damping = 6.0 / 7.0;
		//prm.precond.coarsening.over_interp = 1.0;
		Solver solve(A, prm);
		CSim::TimerMan::timer("Benchmark/solve/build_levels").stop();

		// Show the mini-report on the constructed solver:
		std::cout << solve << std::endl;

		// Solve the system with the zero initial approximation:
		size_t iters;
		float error;

		CSim::TimerMan::timer("Benchmark/solve/solve").start();
		std::tie(iters, error) = solve(A, vector_rhs, vector_x);
		std::cout << "iters: " << iters << "\terror: " << error << '\n';
		CSim::TimerMan::timer("Benchmark/solve/solve").stop();
	}
	else if (smoother == 3) {
		typedef amgcl::make_solver<
			amgcl::amg<
			PBackend,
			amgcl::coarsening::smoothed_aggregation,
			//amgcl::relaxation::damped_jacobi
			amgcl::relaxation::spai0
			>,
			amgcl::solver::cg<SBackend>
		> Solver;

		// Initialize the solver with the system matrix:
		Solver::params prm;
		prm.solver.tol = 1e-7;
		//prm.precond.coarsening.over_interp = 1.0;
		Solver solve(A, prm);
		CSim::TimerMan::timer("Benchmark/solve/build_levels").stop();

		// Show the mini-report on the constructed solver:
		std::cout << solve << std::endl;

		// Solve the system with the zero initial approximation:
		size_t iters;
		float error;

		CSim::TimerMan::timer("Benchmark/solve/solve").start();
		std::tie(iters, error) = solve(A, vector_rhs, vector_x);
		std::cout << "iters: " << iters << "\terror: " << error << '\n';
		CSim::TimerMan::timer("Benchmark/solve/solve").stop();
	}
	else {
		printf("selected smoothing method not implemented\n");
		exit(0);
	}
}

void solve_viscosity_UAAMG(	packed_FloatGrid3 initial_guess, simd_uaamg::simd_viscosity3d& simd_solver) {


	//convert each level to a sparse matrix
	std::vector<Eigen::SparseMatrix<float, Eigen::RowMajor>> LHS_matrix, restriction_matrix, prolongation_matrix;
	std::vector<std::vector<int>> color_update_mask;

	//generate finest level matrix
	CSim::TimerMan::timer("Benchmark/solve/finest_matrix").start();
	std::vector<Eigen::Triplet<float>> lv0_LHS_triplet;
	CSim::TimerMan::timer("Benchmark/solve/finest_matrix/get_triplet").start();
	simd_solver.m_matrix_levels[0]->get_triplets(lv0_LHS_triplet);
	CSim::TimerMan::timer("Benchmark/solve/finest_matrix/get_triplet").stop();
	int level0_dof = simd_solver.m_matrix_levels[0]->m_ndof;
	CSim::TimerMan::timer("Benchmark/solve/finest_matrix/build_matrix").start();
	LHS_matrix.push_back(Eigen::SparseMatrix<float, Eigen::RowMajor>{});
	LHS_matrix.back().resize(level0_dof, level0_dof);
	LHS_matrix.back().setFromTriplets(lv0_LHS_triplet.begin(), lv0_LHS_triplet.end());
	CSim::TimerMan::timer("Benchmark/solve/finest_matrix/build_matrix").stop();
	CSim::TimerMan::timer("Benchmark/solve/finest_matrix").stop();

	CSim::TimerMan::timer("Benchmark/solve/build_levels").start();
	for (int i = 0; i < simd_solver.m_matrix_levels.size() - 1; i++) {
		CSim::TimerMan::timer("Benchmark/solve/build_levels/lv" + std::to_string(i)).start();
		//generate the triplets for this level of matrix
		int this_level_dof = simd_solver.m_matrix_levels[i]->m_ndof;

		color_update_mask.push_back(std::vector<int>{});
		color_update_mask.back().resize(this_level_dof);

		CSim::TimerMan::timer("Benchmark/solve/build_levels/lv" + std::to_string(i) + "/colormask").start();
		for (int c = 0; c < 3; c++) {
			auto color_update_mask_setter = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index32) {
				for (auto iter = leaf.beginValueOn(); iter; ++iter) {
					auto gcoord = iter.getCoord();
					auto sum = gcoord.x() + gcoord.y() + gcoord.z();
					if (sum % 2 == 0) {
						color_update_mask.back()[iter.getValue()] = c*2;
					}
					else {
						color_update_mask.back()[iter.getValue()] = 1 + c*2;
					}
				}
			};
			simd_solver.m_matrix_levels[i]->m_dof_manager[c]->foreach(color_update_mask_setter);
		}
		CSim::TimerMan::timer("Benchmark/solve/build_levels/lv" + std::to_string(i) + "/colormask").stop();

		//generate triplets for restriction and prolongation matrix
		int child_level = i + 1;
		if (child_level < simd_solver.m_matrix_levels.size()) {
			auto child_laplacian = simd_solver.m_matrix_levels[child_level].get();
			std::vector<Eigen::Triplet<float>> triplets;
			for (int c = 0; c < 3; c++) {
				auto child_dofidx = child_laplacian->m_velocity_DOF.idx[c];
				//get all child nodes
				std::vector<openvdb::Int32Tree::LeafNodeType*> child_nodes;
				child_nodes.reserve(child_dofidx->tree().leafCount());
				child_dofidx->tree().getNodes(child_nodes);

				size_t nchild_nodes = child_nodes.size();
				restriction_triplet_reducer triplet_reducer(child_nodes, simd_solver.m_matrix_levels[i]->m_velocity_DOF.idx[c]);
				//CSim::TimerMan::timer("Benchmark/solve/cvt/lv" + std::to_string(i) + "/Rtriplet").start();
				tbb::parallel_reduce(tbb::blocked_range<size_t>(0, nchild_nodes), triplet_reducer);
				//CSim::TimerMan::timer("Benchmark/solve/cvt/lv" + std::to_string(i) + "/Rtriplet").stop();
				auto begin = triplets.size();
				triplets.resize(begin + triplet_reducer.m_triplets.size());
				std::copy(triplet_reducer.m_triplets.begin(), triplet_reducer.m_triplets.end(), triplets.data() +  begin);
			}
			CSim::TimerMan::timer("Benchmark/solve/build_levels/lv" + std::to_string(i) + "/Rbuild").start();
			restriction_matrix.push_back(Eigen::SparseMatrix<float, Eigen::RowMajor>{});
			restriction_matrix.back().resize(child_laplacian->m_ndof, this_level_dof);
			restriction_matrix.back().setFromTriplets(triplets.begin(), triplets.end());
			CSim::TimerMan::timer("Benchmark/solve/build_levels/lv" + std::to_string(i) + "/Rbuild").stop();
			CSim::TimerMan::timer("Benchmark/solve/build_levels/lv" + std::to_string(i) + "/Pbuild").start();
			prolongation_matrix.push_back(restriction_matrix.back().transpose() * 8.0f);
			CSim::TimerMan::timer("Benchmark/solve/build_levels/lv" + std::to_string(i) + "/Pbuild").stop();
		}
		CSim::TimerMan::timer("Benchmark/solve/build_levels/lv" + std::to_string(i) + "/RAP").start();
		LHS_matrix.push_back(restriction_matrix.back() * LHS_matrix.back() * prolongation_matrix.back());
		CSim::TimerMan::timer("Benchmark/solve/build_levels/lv" + std::to_string(i) + "/RAP").stop();

		CSim::TimerMan::timer("Benchmark/solve/build_levels/lv" + std::to_string(i)).stop();
	}
	CSim::TimerMan::timer("Benchmark/solve/build_levels").stop();
	//std::vector<float> vector_rhs, vector_x;
	Eigen::VectorXf eigen_rhs, eigen_x;

	//right hand side, initial guess
	simd_solver.m_matrix_levels[0]->to_Eigenvector(eigen_rhs, simd_solver.m_rhs);
	simd_solver.m_matrix_levels[0]->to_Eigenvector(eigen_x, initial_guess);


	explicit_PCG_solver UAAMGPCG(LHS_matrix, restriction_matrix, prolongation_matrix, color_update_mask, 6);
	UAAMGPCG.m_sor_coef = 1.0;
	UAAMGPCG.m_smooth_count = 2;
	UAAMGPCG.mu = 2;
	//UAAMGPCG.mSmoother = explicit_PCG_solver::SmootherOption::RedBlackGaussSeidel;

	int smoother = YamlSingleton::get()["viscosity_smoother"].as<int>();
	switch (smoother) {
	case 0:
		//damped jacobi
		UAAMGPCG.m_smoother = explicit_PCG_solver::Smoother::DAMPED_JACOBI;
		UAAMGPCG.m_damped_jacobi_coef = 0.72;
		break;
	case 1:
		UAAMGPCG.m_smoother = explicit_PCG_solver::Smoother::SCHEDULED_RELAXED_JACOBI;
		break;
	case 2:
		UAAMGPCG.m_smoother = explicit_PCG_solver::Smoother::RED_BLACK_GAUSS_SEIDEL;
		break;
	case 3:
		UAAMGPCG.m_smoother = explicit_PCG_solver::Smoother::SPAI0;
		break;
	}
	
	CSim::TimerMan::timer("Benchmark/solve/solve").start();
	UAAMGPCG.pcg_solve(eigen_x, eigen_rhs);
	CSim::TimerMan::timer("Benchmark/solve/solve").stop();
}

void benchmark_viscosity() {

	CSim::TimerMan::timer("Benchmark/SDF_setup").start();
	float radius = 0.5;
	int nx = YamlSingleton::get()["nx"].as<int>();
	float dx = 1.0f / nx;
	auto liquid_sdf = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(radius, openvdb::Vec3f(0.f), dx);
	liquid_sdf->setName("m_liquid_sdf");

	sopFillSDF(*liquid_sdf, 0);
	liquid_sdf->tree().voxelizeActiveTiles();

	//openvdb::io::File("m_liquid_sdf.vdb").write({ liquid_sdf });

	//openvdb::tools::dilateActiveValues(liquid_sdf->tree());
	//liquid_sdf->tree().voxelizeActiveTiles();

	auto solid_sdf = openvdb::createLevelSet<openvdb::FloatGrid>(dx);
	solid_sdf = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(radius, openvdb::Vec3f(0, -radius * 3.0f / 4.0f, 0), dx);
	solid_sdf->setName("m_solid_sdf");
	float viscosity_coef = YamlSingleton::get()["viscosity_coef"].as<float>();
	auto viscosity = openvdb::FloatGrid::create(viscosity_coef);
	viscosity->setName("m_viscosity");
	//L_with_level lv0(viscosity, liquid_sdf, solid_sdf, 1, 1);
	//lv0.IO("dbg.vdb");
	//unit_test_stencil_pattern(lv0);
	//unit_test_coarsening_stencil_pattern(lv0);
	//unit_test_RBGS_pattern(lv0);
	CSim::TimerMan::timer("Benchmark/SDF_setup").stop();

	CSim::TimerMan::timer("Benchmark/random_rhs").start();
	packed_FloatGrid3 liquid_velocity;
	//set the initial velocity to follow the liquid sdf, but with randomness
	for (int i = 0; i < 3; i++) {
		liquid_velocity.v[i]->setTree(
			std::make_shared<openvdb::FloatTree>(liquid_sdf->tree(), 0.f, openvdb::TopologyCopy()));
		auto treeman = openvdb::tree::LeafManager<openvdb::FloatTree>(liquid_velocity.v[i]->tree());
		float nleaf = treeman.leafCount();
		auto random_setter = [&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index leafpos) {
			std::random_device device;
			std::mt19937 generator(/*seed=*/leafpos*(i+1));
			std::uniform_real_distribution<> distribution(-0.5, 0.5);
			for (auto iter = leaf.beginValueAll(); iter; ++iter) {
				//if (i == 1) {
				//	iter.setValue(1);
				//}
				//else {
				//	iter.setValue(0);
				//}
				iter.setValue(iter.getValue() + distribution(generator) + (iter.offset() - 256.f) * (1.0f / 512.0f) + (leafpos - nleaf / 2) / nleaf);
				//iter.setValue(0);
				iter.setValueOn();
			}
		};
		treeman.foreach(random_setter);
	}
	CSim::TimerMan::timer("Benchmark/random_rhs").stop();

	CSim::TimerMan::timer("Benchmark/create_viscosity_matrix").start();
	openvdb::Vec3fGrid::Ptr solid_velocity = openvdb::Vec3fGrid::create(openvdb::Vec3f(0.f));
	auto level0 = simd_uaamg::L_with_level::create_viscosity_matrix(
		viscosity,
		liquid_sdf,
		solid_sdf, 1.0f / 24.0f, 1000.0f);
	CSim::TimerMan::timer("Benchmark/create_viscosity_matrix").stop();

	CSim::TimerMan::timer("Benchmark/build_levels_simd").start();
	simd_uaamg::simd_viscosity3d simdsolver(level0,
		liquid_velocity,
		solid_velocity);
	auto rhs = simdsolver.m_rhs;
	CSim::TimerMan::timer("Benchmark/build_levels_simd").stop();

	packed_FloatGrid3 init_guess = simdsolver.m_matrix_levels[0]->get_zero_vec();

	int solver_method = YamlSingleton::get()["viscosity_solver_method"].as<int>();

	CSim::TimerMan::timer("Benchmark/solve").start();
	switch (solver_method) {
	case 0:
		//Eigen Jacobi CG
		solve_viscosity_Eigen_diagonalPCG(rhs, init_guess, simdsolver.m_matrix_levels[0].get());
		break;
	case 1:
		//Eigen ICPCG
		//solve_viscosity_Eigen_ICPCG(rhs, init_guess, level0);
		break;
	case 2:
		//Batty ICPCG
		solve_viscosity_Batty_ICPCG(rhs, init_guess, simdsolver.m_matrix_levels[0].get());
		break;
	case 3:
		//UAAMG
		solve_viscosity_UAAMG(init_guess, simdsolver);
		break;
	case 4:
	default:
		//SIMD-UAAMG
		simdsolver.pcg_solve(init_guess, 1e-7);
		break;
	case 5:
		//AMGCL
		solve_viscosity_amgcl(rhs, init_guess, simdsolver.m_matrix_levels[0].get());
		break;
	}

	std::string output_prefix = YamlSingleton::get()["output_prefix"].as<std::string>();
	std::filesystem::path poisson_output_directory(output_prefix);
	std::filesystem::create_directory(poisson_output_directory);
	poisson_output_directory /= ("viscosity_" + std::to_string(nx));
	std::filesystem::create_directory(poisson_output_directory);

	std::filesystem::path output_vdb = poisson_output_directory / "sdf.vdb";

	printf("solve done\n");
	CSim::TimerMan::timer("Benchmark/solve").stop();
	CSim::TimerMan::timer("Benchmark/IO").start();
	openvdb::io::File(output_vdb.string()).write({ liquid_sdf, solid_sdf, init_guess.v[0],init_guess.v[1], init_guess.v[2] });
	CSim::TimerMan::timer("Benchmark/IO").stop();
}


int main(int argc, char** argv) {
	if (argc != 2) {
		printf("Usage: CG_benchmark configuration_file_name.yaml\n");
		exit(0);
	}

	//try to open the file
	FILE* pFile = fopen(argv[1], "r");
	if (pFile == nullptr) {
		printf("configuration file %s does not exist\n", argv[1]);
		exit(0);
	}
	fclose(pFile);

	YamlSingleton::loadfile(argv[1]);
	CSim::TimerMan::timer("Benchmark").start();
	if (YamlSingleton::get()["equation_type"].as<std::string>() == "pressure") {
		benchmark_poisson();
	}
	if (YamlSingleton::get()["equation_type"].as<std::string>() == "viscosity") {
		benchmark_viscosity();
	}
	CSim::TimerMan::timer("Benchmark").stop();
	std::cout << "Benchmark time " << CSim::TimerMan::timer("Benchmark").lasttime() << std::endl;
	return 0;
}