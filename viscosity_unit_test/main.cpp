#include <iostream>
#include "tbb/tbbmalloc_proxy.h"
#include "openvdb/tools/LevelSetSphere.h"
#include "openvdb/tools/Morphology.h"
#include "Tracy.hpp"
#include "simd_viscosity3d.h"

void unit_test_stencil_pattern(simd_uaamg::L_with_level& level) {
	
	auto scratchpad = level.get_zero_vec();
	
	scratchpad.v[2]->tree().setValue(openvdb::Coord(0, 0, 0), 1);
	
	Eigen::VectorXf lhs = level.to_Eigenvector(scratchpad);
	level.build_explicit_matrix();
	Eigen::VectorXf rhs = level.m_explicit_matrix * lhs;
	
	level.write_to_FloatGrid3(scratchpad, rhs);

	openvdb::io::File("stencil.vdb").write({ scratchpad.v[0], scratchpad.v[1], scratchpad.v[2] });
}

void unit_test_coarsening_stencil_pattern(const simd_uaamg::L_with_level& level) {
	simd_uaamg::L_with_level coarse_level(level, simd_uaamg::L_with_level::Coarsening());

	auto scratchpad = coarse_level.get_zero_vec();
	scratchpad.v[1]->tree().setValue(openvdb::Coord(1, 0, -1), 1);
	Eigen::VectorXf lhs = coarse_level.to_Eigenvector(scratchpad);
	coarse_level.build_explicit_matrix();
	Eigen::VectorXf rhs = coarse_level.m_explicit_matrix * lhs;
	coarse_level.write_to_FloatGrid3(scratchpad, rhs);
	coarse_level.IO("dbglv" + std::to_string(coarse_level.m_level) + ".vdb");
	openvdb::io::File("stencil_lv"+std::to_string(coarse_level.m_level)+".vdb").write({ scratchpad.v[0], scratchpad.v[1], scratchpad.v[2] });
}

void unit_test_simd_apply_operator(simd_uaamg::L_with_level& level) {
	auto scratchpad = level.get_zero_vec();
	scratchpad.v[1]->tree().setValue(openvdb::Coord(1, 0, -1), 1);
	auto result = level.get_zero_vec();
	auto op = level.get_light_weight_applier(result, scratchpad, result, simd_uaamg::L_with_level::working_mode::NORMAL);
	level.run(op);

	openvdb::io::File("simdstencil.vdb").write({ result.v[0], result.v[1], result.v[2] });
}

void unit_test_rhs_and_residual(simd_uaamg::L_with_level& level) {
	Eigen::VectorXf all1;
	all1.setOnes(level.m_ndof);
	
	auto all_one_vel = level.to_packed_FloatGrid3(all1);
	auto solidvel = openvdb::Vec3fGrid::create(openvdb::Vec3f(0.f));
	
	auto rhs = level.build_rhs(all_one_vel, solidvel);

	auto residual = level.get_zero_vec();
	level.residual_apply(residual, all_one_vel, rhs);
	residual.setName("residual");

	openvdb::io::File("rhs.vdb").write({ rhs.v[0], rhs.v[1], rhs.v[2] ,
		residual.v[0], residual.v[1], residual.v[2] });
}

void unit_test_smoothing(simd_uaamg::L_with_level& level) {
	Eigen::VectorXf initial_guess;
	initial_guess.setRandom(level.m_ndof);

	float invndof = 1.0f / level.m_ndof;
	for (int i = 0; i < level.m_ndof; i++) {
		initial_guess[i] += float(i-level.m_ndof/2) * invndof;
	}

	auto rhs_grid = level.get_zero_vec();
	auto scratchpad = level.get_zero_vec();
	auto lhs_grid = level.to_packed_FloatGrid3(initial_guess);


	//rhs_grid.v[0]->tree().setValue(openvdb::Coord(0, 0, 0), 1000);

	int i = 0;
	for (; i < 50; i++) {
		openvdb::io::File("lhs" + std::to_string(i) + ".vdb").write({ lhs_grid.v[0], lhs_grid.v[1], lhs_grid.v[2] });
		for (int j = 0; j < 1; j++) {
			level.XYZ_RBGS_apply(scratchpad, lhs_grid, rhs_grid);
		}
		//level.ZYX_Jacobi_apply(lhs_grid, scratchpad, rhs_grid);
		//scratchpad.swap(lhs_grid);
	}
	openvdb::io::File("lhs" + std::to_string(i) + ".vdb").write({ lhs_grid.v[0], lhs_grid.v[1], lhs_grid.v[2] });
}

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
				iter.setValue(iter.getValue() + distribution(generator) +(iter.offset() - 256.f) * (1.0f / 512.0f) + (leafpos - nleaf / 2) / nleaf);
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

void unit_test_RBGS_pattern(simd_uaamg::L_with_level & lv) {
	//given a random initial
	//smooth the result to zero
	Eigen::VectorXf randomvec;
	randomvec.setRandom(lv.m_ndof);

	auto initial_lhs = lv.to_packed_FloatGrid3(randomvec);

	auto zero_rhs = lv.get_zero_vec();
	auto scratch_pad = initial_lhs.deepCopy();

	auto residual_pad = zero_rhs.deepCopy();

	Eigen::VectorXf zerovec;
	zerovec.setZero(lv.m_ndof);

	int niter = 50;
	for (int i = 0; i < niter; ++i) {
		for (int channel = 0; channel <1; channel++) {

			//redGS
			{
				lv.set_grid_to_zero(residual_pad);

				auto ROP = lv.get_light_weight_applier(initial_lhs, initial_lhs, zero_rhs, simd_uaamg::L_with_level::working_mode::RED_GS);
				ROP.set_channel(channel);
				lv.m_dof_manager[channel]->foreach(ROP);

				/*lv.set_grid_to_zero(residual_pad);
				auto residual_op = lv.get_light_weight_applier(residual_pad, initial_lhs, zero_rhs, L_with_level::working_mode::RESIDUAL);
				residual_op.set_channel(channel);
				lv.m_dof_manager[channel]->foreach(residual_op);
				openvdb::io::File("red_residual_before_updatec" + std::to_string(channel) + "frame" + std::to_string(i) + ".vdb")
					.write({ residual_pad.v[channel] });*/

				//scratchpad is updated in red points in channel
				//calculate the residual with scratchpad as lhs should have zero red residual
				//lv.set_grid_to_zero(residual_pad);
				auto residual_op = lv.get_light_weight_applier(residual_pad, initial_lhs, zero_rhs, simd_uaamg::L_with_level::working_mode::RESIDUAL);
				residual_op.set_channel(channel);
				lv.m_dof_manager[channel]->foreach(residual_op);
				openvdb::io::File("red_residualc" + std::to_string(channel) + "frame" + std::to_string(i) + ".vdb")
					.write({ residual_pad.v[channel] });


				////check if the matrix free residual is the same as the explicit matrix residual
				//Eigen::VectorXf eigenresidual;
				//Eigen::VectorXf eigenlhs;
				//lv.to_Eigenvector(eigenlhs, scratch_pad);
				//eigenresidual = zerovec - lv.m_explicit_matrix * eigenlhs;
				//auto explicit_residual_grid = lv.to_packed_FloatGrid3(eigenresidual);
				//openvdb::io::File("red_explicit_residualc" + std::to_string(channel) + "frame" + std::to_string(i) + ".vdb")
				//	.write({ explicit_residual_grid.v[channel] });

				/*auto red_update = scratch_pad.v[channel]->deepCopy();

				auto sub = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) {
					openvdb::FloatTree::LeafNodeType* original_val_leaf, *new_val_leaf, *update_val_leaf;
					original_val_leaf = initial_lhs.v[channel]->tree().probeLeaf(leaf.origin());
					new_val_leaf = scratch_pad.v[channel]->tree().probeLeaf(leaf.origin());
					update_val_leaf = red_update->tree().probeLeaf(leaf.origin());
					for (auto iter = leaf.beginValueOn(); iter; ++iter) {
						update_val_leaf->setValueOnly(iter.offset(), new_val_leaf->getValue(iter.offset()) - original_val_leaf->getValue(iter.offset()));
					}
				};
				lv.m_dof_manager[channel]->foreach(sub);
				openvdb::io::File("red_updatec" + std::to_string(channel) + "frame" + std::to_string(i) + ".vdb")
					.write({ red_update });*/
			}

			//blackGS
			{
				lv.set_grid_to_zero(residual_pad);

				auto BOP = lv.get_light_weight_applier(initial_lhs, initial_lhs, zero_rhs, simd_uaamg::L_with_level::working_mode::BLACK_GS);
				BOP.set_channel(channel);
				lv.m_dof_manager[channel]->foreach(BOP);
				/*lv.set_grid_to_zero(residual_pad);
				auto residual_op = lv.get_light_weight_applier(residual_pad, initial_lhs, zero_rhs, L_with_level::working_mode::RESIDUAL);
				residual_op.set_channel(channel);
				lv.m_dof_manager[channel]->foreach(residual_op);
				openvdb::io::File("black_residual_before_updatec" + std::to_string(channel) + "frame" + std::to_string(i) + ".vdb")
					.write({ residual_pad.v[channel] });*/

				//lv.set_grid_to_zero(residual_pad);
				//initial_lhs is updated in black points in channel
				//calculate the residual with initial_lhs as lhs should have zero black residual
				auto residual_op = lv.get_light_weight_applier(residual_pad, initial_lhs, zero_rhs, simd_uaamg::L_with_level::working_mode::RESIDUAL);
				residual_op.set_channel(channel);
				lv.m_dof_manager[channel]->foreach(residual_op);
				openvdb::io::File("black_residualc" + std::to_string(channel) + "frame" + std::to_string(i) + ".vdb")
					.write({ residual_pad.v[channel] });

				//auto black_update = initial_lhs.v[channel]->deepCopy();

				/*auto sub = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) {
					openvdb::FloatTree::LeafNodeType* original_val_leaf, * new_val_leaf, * update_val_leaf;
					original_val_leaf = scratch_pad.v[channel]->tree().probeLeaf(leaf.origin());
					new_val_leaf = initial_lhs.v[channel]->tree().probeLeaf(leaf.origin());
					update_val_leaf = black_update->tree().probeLeaf(leaf.origin());
					for (auto iter = leaf.beginValueOn(); iter; ++iter) {
						update_val_leaf->setValueOnly(iter.offset(), new_val_leaf->getValue(iter.offset()) - original_val_leaf->getValue(iter.offset()));
					}
				};
				lv.m_dof_manager[channel]->foreach(sub);
				openvdb::io::File("black_updatec" + std::to_string(channel) + "frame" + std::to_string(i) + ".vdb")
					.write({ black_update });*/
			}
		}
	}
}

int main() {
	float radius = 10;
	float dx = 1.0f/16.0f;
	auto liquid_sdf = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(radius, openvdb::Vec3f(0.f), dx);
	liquid_sdf->setName("m_liquid_sdf");

	sopFillSDF(*liquid_sdf, 0);
	liquid_sdf->tree().voxelizeActiveTiles();

	//openvdb::io::File("m_liquid_sdf.vdb").write({ liquid_sdf });

	//openvdb::tools::dilateActiveValues(liquid_sdf->tree());
	//liquid_sdf->tree().voxelizeActiveTiles();

	auto solid_sdf = openvdb::createLevelSet<openvdb::FloatGrid>(dx);
	solid_sdf = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(radius, openvdb::Vec3f(0, -radius*3.0f/4.0f, 0), dx);
	solid_sdf->setName("m_solid_sdf");
	auto viscosity = openvdb::FloatGrid::create(50);
	viscosity->setName("m_viscosity");
	//L_with_level lv0(viscosity, liquid_sdf, solid_sdf, 1, 1);
	//lv0.IO("dbg.vdb");
	//unit_test_stencil_pattern(lv0);
	//unit_test_coarsening_stencil_pattern(lv0);
	//unit_test_RBGS_pattern(lv0);
	unit_test_PCG(viscosity, liquid_sdf, solid_sdf);

	printf("hello\n");
}