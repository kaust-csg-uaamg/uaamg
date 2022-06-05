#pragma once
//FLIP simulation based on openvdb points and grid structure
//#include "openvdb/openvdb.h"
#include "openvdb/points/PointConversion.h"
#include "sdf_vel_pair.h"
#include "packed3grids.h"

class FLIP_Solver_OpenVDB;
struct Compressed_FLIP_Object_OpenVDB;

//A FLIP Object consists of OpenVDB points, as well as a velocity field
class FLIP_Object_OpenVDB {
public:
	//Attribute encoding types
	using PositionCodec = openvdb::points::FixedPointCodec</*one byte*/false>;
	//using PositionCodec = openvdb::points::TruncateCodec;
	//using PositionCodec = openvdb::points::NullCodec;
	using PositionAttribute = openvdb::points::TypedAttributeArray<openvdb::Vec3f, PositionCodec>;
	using VelocityCodec = openvdb::points::TruncateCodec;
	//using VelocityCodec = openvdb::points::NullCodec;
	using VelocityAttribute = openvdb::points::TypedAttributeArray<openvdb::Vec3f, VelocityCodec>;
public:
	FLIP_Object_OpenVDB(float in_dx);
	FLIP_Object_OpenVDB(const sdf_vel_pair& in_sdfvel, float in_dx);
	FLIP_Object_OpenVDB(const Compressed_FLIP_Object_OpenVDB& in_compressed_object);

	void set_viscosity(const float in_viscosity) { m_viscosity = in_viscosity; }
	void set_density(const float in_density) { m_density = in_density; }
	openvdb::points::PointDataGrid::Ptr get_particles() { return m_particles; }

protected:
	friend class FLIP_Solver_OpenVDB;
	friend class Compressed_FLIP_Object_OpenVDB;
	//The PointDataGrid sorts particles in voxels with lattice in center.
	//It contains position and velocity attributes.
	//The position attribute is with respect to voxel center with range(-0.5,0.5)
	//The velocity attribute has world-coordinate units in m/s
	openvdb::points::PointDataGrid::Ptr m_particles;

	//Three separate velocity channels
	//This is the velocity field that would be used for advecting the particles
	packed_FloatGrid3 m_velocity;

	float m_viscosity{ 0.f };
	float m_density{ 1000.f };

	//Radius of the particle.
	//It is prefered to be set as 0.5*sqrt(3)*dx,
	//so that when one particle is inside a voxel,
	//it will make the liquid sdf in the center negative.
	float m_particle_radius;
};

//Compressed FLIP Object that is economic for storage
struct Compressed_FLIP_Object_OpenVDB {
	Compressed_FLIP_Object_OpenVDB(const FLIP_Object_OpenVDB& in_object);
	void write(std::string filename) const;

	static void narrow_band_particles(
		openvdb::points::PointDataGrid::Ptr out_nb_particles,
		openvdb::FloatGrid::Ptr out_interior_sdf,
		openvdb::points::PointDataGrid::Ptr in_particles, int nlayer = 3);

	openvdb::points::PointDataGrid::Ptr surface_layer_particles;
	openvdb::Vec3fGrid::Ptr velocity;
	openvdb::BoolGrid::Ptr interior_mask;
};

//A FLIP_Solver_OpenVDB evolves the state of a FLIP Object forward in time
class FLIP_Solver_OpenVDB {
public:
	//Attributes encoding types
	using PositionCodec = FLIP_Object_OpenVDB::PositionCodec;
	using PositionAttribute = FLIP_Object_OpenVDB::PositionAttribute;
	using VelocityCodec = FLIP_Object_OpenVDB::VelocityCodec;
	using VelocityAttribute = FLIP_Object_OpenVDB::VelocityAttribute;

	enum class DomainBoundaryType {
		SINK,
		COLLISION,
		KINEMATIC_SOLID
	};

	enum class SimulationMethod {
		ADVECTION_PROJECTION,
		ADVECTION_REFLECITON
	};

	enum class ViscosityOption {
		OFF,
		ON
	};

	enum class ParticleIntegrationMethod : int {
		FORWARD_EULER=1,
		RK2=2,
		RK3=3,
		RK4=4
	};

	struct SolverParameters {
		//Default constructor with some default parameters
		SolverParameters();
		SolverParameters(const SolverParameters& other) = default;
		SolverParameters& operator=(const SolverParameters& other) = default;
		openvdb::Vec3f domain_min, domain_max;
		float cfl;
		float PIC_component;
		SimulationMethod simulation_method;
		DomainBoundaryType domain_boundary_type;
		ViscosityOption enable_viscosity;
		ParticleIntegrationMethod particle_integration_method;
		float viscosity_solver_accuracy;
		float pressure_solver_accuracy;
		float boundary_friction_coefficient;
		openvdb::Vec3f gravity;
	};

	struct SolverStats {

		void init() {
			auto zerotime = std::chrono::duration<double>::zero();
			advection_time = zerotime;
			p2g_time = zerotime;
			emission_time = zerotime;
			sink_time = zerotime;
			update_solid_sdf_vel_time = zerotime;
			extrapolate_time = zerotime;
			substep_total_time = zerotime;
			particle_count = 0;

			
			//poisson
			poisson_time_total = zerotime;
			poisson_time_build = zerotime;
			poisson_time_solve = zerotime;
			poisson_pcg_iterations = 0;
			poisson_MG_iterations = 0;
			poisson_pcg_failed = false;
			poisson_ndof = 0;

			//viscosity
			viscosity_time_total = zerotime;
			viscosity_time_build = zerotime;
			viscosity_time_solve = zerotime;
			viscosity_iterations = 0;
			viscosity_ndof = 0;
		}

		//FLIP stages time
		std::chrono::duration<double> advection_time, 
			p2g_time, emission_time, sink_time, 
			update_solid_sdf_vel_time, extrapolate_time, substep_total_time;
		int particle_count;

		//Pressure Poisson's Equation
		std::chrono::duration<double> poisson_time_total, poisson_time_build, poisson_time_solve;
		int poisson_pcg_iterations, poisson_MG_iterations;
		bool poisson_pcg_failed;
		int poisson_ndof;

		//Viscosity Solver
		std::chrono::duration<double> viscosity_time_total, viscosity_time_build, viscosity_time_solve;
		int viscosity_iterations;
		int viscosity_ndof;
	};

	SolverStats m_substep_statistics;
public:
	//public methods
	FLIP_Solver_OpenVDB(const FLIP_Object_OpenVDB& in_object, const FLIP_Solver_OpenVDB::SolverParameters& in_parm);
	~FLIP_Solver_OpenVDB();

	void solve(FLIP_Object_OpenVDB& in_out_object, float in_dt);

	//The bounding box solid sdf
	openvdb::FloatGrid::Ptr get_domain_solid_sdf() { return m_domain_solid_sdf; }

	static void custom_move_points_and_set_flip_vel(
		openvdb::points::PointDataGrid::Ptr in_out_points,
		packed_FloatGrid3 advector,
		packed_FloatGrid3 carried_velocity,
		packed_FloatGrid3 old_velocity,
		openvdb::FloatGrid::Ptr in_solid_sdf,
		openvdb::Vec3fGrid::Ptr in_solid_vel,
		float PIC_component, float dt, int RK_order);

	static void emit_liquid(openvdb::points::PointDataGrid::Ptr in_out_particles,
		const sdf_vel_pair& in_sdf_vel,
		const openvdb::Vec3f& in_emit_world_min,
		const openvdb::Vec3f& in_emit_world_max);

	//particle radius must be smaller than 1.5 dx, because this function
	//only writes to the 3x3x3 voxels around a particle voxel.
	static void particle_to_grid(packed_FloatGrid3 out_velocity, 
		openvdb::FloatGrid::Ptr out_liquid_sdf,
		openvdb::points::PointDataGrid::Ptr in_particles, const float in_particle_radius);

	static openvdb::Vec3fGrid::Ptr calculate_face_weights(openvdb::FloatGrid::Ptr in_liquid_sdf, openvdb::FloatGrid::Ptr in_solid_sdf);

	const SolverParameters& param() { return m_solver_parameters; }
	float cfl();
	float particle_cfl(openvdb::points::PointDataGrid::Ptr in_particles);
	void substep(FLIP_Object_OpenVDB& in_out_object, float substep_dt);
private:
	//private methods
	void init_grids();
	void init_transforms();
	void init_domain_solid_sdf();

	void substep_advection_projection(FLIP_Object_OpenVDB& in_out_object, float substep_dt);
	void substep_advection_reflection_order1(FLIP_Object_OpenVDB& in_out_object, float substep_dt);
	void substep_advection_projection_with_viscosity(FLIP_Object_OpenVDB& in_out_object, float substep_dt);

	void apply_body_force(float in_dt);
	void apply_pressure_gradient(float in_dt, bool reflection = false);
	void calculate_face_weights();
	void extrapolate_velocity(int layer = 3);
	
	void immerse_liquid_phi_in_solids();
	void sink_liquid(openvdb::points::PointDataGrid::Ptr in_out_particles, const std::vector<openvdb::FloatGrid::Ptr>& in_sdfs);
	void solve_pressure_simd(float in_dt, bool reflection = false);
	void solve_viscosity(FLIP_Object_OpenVDB& in_out_object, float in_dt);
	void update_solid_sdf_vel(openvdb::points::PointDataGrid::Ptr in_out_particles, const std::vector<sdf_vel_pair>& in_sdf_vels);
public:
	//for collision in advection step
	std::vector<sdf_vel_pair> m_collision_sdfvel;
	//for emitting liquid
	std::vector<sdf_vel_pair> m_source_sdfvel;
	//for regions removing liquid
	std::vector<openvdb::FloatGrid::Ptr> m_sink_sdfs;
	//for the pressure equation
	std::vector<sdf_vel_pair> m_solid_sdfvel;
private:
	//Parameters of the solver
	FLIP_Solver_OpenVDB::SolverParameters m_solver_parameters;

public:
	//Intermediate variables used in the process of solving

	//length of the voxel
	float m_dx;

	//Signed minimum and maximum voxel index for the simulation
	//it describes the pressure index
	openvdb::Vec3i m_domain_index_begin, m_domain_index_end;

	//The bounding box solid sdf
	openvdb::FloatGrid::Ptr m_domain_solid_sdf;

	//Staggered grid velocities with the same transforms as particles.
	//The origin of the grid is (0,0,0) in world coordinate.
	//That origin is interpreted as the origin of the pressure.
	//Separately store three channels of velocity.
	packed_FloatGrid3 m_packed_velocity, 
		m_packed_velocity_update,
		m_packed_vel_after_p2g, 
		m_packed_velocity_reflected, 
		m_packed_velocity_for_pressure,
		m_solved_viscosity_velocity;

	//Staggered solid velocity with the same transform as velocity
	openvdb::Vec3fGrid::Ptr m_solid_velocity;

	//Solid signed distance function.
	//The origin of m_solid_sdf is (-dx/2,-dx/2,-dx/2) where dx is the voxel
	//edge length
	openvdb::FloatGrid::Ptr m_solid_sdf;

	//Liquid signed distance function calculated by p2g.
	//It has the same transform as the m_particles, pressure
	openvdb::FloatGrid::Ptr m_liquid_sdf;

	//For each face of the voxel, calculate the face weight used for variational pressure solver
	openvdb::Vec3fGrid::Ptr m_face_weight;

	//Pressure grid with the same transform as the m_particles in the FLIP_Object_OpenVDB.
	//In that way, each m_particles voxel coincide with a fluid simulation voxel.
	openvdb::FloatGrid::Ptr m_pressure;
	//Right hand side used for the pressure projection
	openvdb::FloatGrid::Ptr m_rhs;

	//Transforms that will be shared
	openvdb::math::Transform::Ptr m_voxel_center_transform, m_voxel_vertex_transform;
};
