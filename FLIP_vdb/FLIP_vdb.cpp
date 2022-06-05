#if WIN32
#if !defined _ENABLE_EXTENDED_ALIGNED_STORAGE
#define _ENABLE_EXTENDED_ALIGNED_STORAGE
#endif
#endif
#include <stdint.h>
#include "tbb/scalable_allocator.h"
#include "tbb/tbb_allocator.h"
#include "tbb/blocked_range3d.h"
#include "FLIP_vdb.h"
#include <openvdb/points/PointCount.h>
#include "openvdb/tree/LeafManager.h"
#include "openvdb/points/PointAdvect.h"
#include "openvdb/tools/Morphology.h"
#include "openvdb/tools/LevelSetTracker.h"
#include "openvdb/tools/MeshToVolume.h"
//#include "SimOptions.h"
#include "levelset_util.h"

#include <atomic>
//intrinsics
#include <xmmintrin.h>
#include "simd_vdb_poisson.h"
#include "simd_viscosity3d.h"
#include "vdb_velocity_extrapolator.h"
#include <thread>

#include "Timer.h"

//Tracy profiler
#include "Tracy.hpp"

template <typename T>
using allocator_type = tbb::tbb_allocator<T>;
template <typename T>
using custom_vector_type = std::vector<T, allocator_type<T>>;

namespace {
struct BoxSampler {
	template < typename axr_type>
	static void get_eight_data(float* data,
		axr_type& axr,
		openvdb::Coord gcoord) {
		auto& i = gcoord[0];
		auto& j = gcoord[1];
		auto& k = gcoord[2];

		data[0] = axr.getValue(gcoord); //000
		k++; data[1] = axr.getValue(gcoord); //001
		j++; data[3] = axr.getValue(gcoord); //011
		k--; data[2] = axr.getValue(gcoord); //010
		i++; data[6] = axr.getValue(gcoord); //110
		k++; data[7] = axr.getValue(gcoord); //111
		j--; data[5] = axr.getValue(gcoord); //101
		k--; data[4] = axr.getValue(gcoord); //100
	}

	template <typename axr_1_type, typename axr_2_type>
	static void get_eight_data2(float* data_1, float* data_2,
		axr_1_type& axr_1, axr_2_type& axr_2,
		openvdb::Coord gcoord) {
		auto& i = gcoord[0];
		auto& j = gcoord[1];
		auto& k = gcoord[2];

		data_1[0] = axr_1.getValue(gcoord); data_2[0] = axr_2.getValue(gcoord);//000
		k++; data_1[1] = axr_1.getValue(gcoord); data_2[1] = axr_2.getValue(gcoord);//001
		j++; data_1[3] = axr_1.getValue(gcoord); data_2[3] = axr_2.getValue(gcoord); //011
		k--; data_1[2] = axr_1.getValue(gcoord); data_2[2] = axr_2.getValue(gcoord);//010
		i++; data_1[6] = axr_1.getValue(gcoord); data_2[6] = axr_2.getValue(gcoord);//110
		k++; data_1[7] = axr_1.getValue(gcoord); data_2[7] = axr_2.getValue(gcoord); //111
		j--; data_1[5] = axr_1.getValue(gcoord); data_2[5] = axr_2.getValue(gcoord);//101
		k--; data_1[4] = axr_1.getValue(gcoord); data_2[4] = axr_2.getValue(gcoord);//100
	}

	template <typename axr_1_type, typename axr_2_type, typename axr_3_type>
	static void get_eight_data3(float* data_1, float* data_2, float* data_3,
		axr_1_type& axr_1, axr_2_type& axr_2, axr_3_type& axr_3,
		openvdb::Coord gcoord) {
		auto& i = gcoord[0];
		auto& j = gcoord[1];
		auto& k = gcoord[2];

		data_1[0] = axr_1.getValue(gcoord); data_2[0] = axr_2.getValue(gcoord); data_3[0] = axr_3.getValue(gcoord);//000
		k++; data_1[1] = axr_1.getValue(gcoord); data_2[1] = axr_2.getValue(gcoord); data_3[1] = axr_3.getValue(gcoord);//001
		j++; data_1[3] = axr_1.getValue(gcoord); data_2[3] = axr_2.getValue(gcoord); data_3[3] = axr_3.getValue(gcoord);//011
		k--; data_1[2] = axr_1.getValue(gcoord); data_2[2] = axr_2.getValue(gcoord); data_3[2] = axr_3.getValue(gcoord);//010
		i++; data_1[6] = axr_1.getValue(gcoord); data_2[6] = axr_2.getValue(gcoord); data_3[6] = axr_3.getValue(gcoord);//110
		k++; data_1[7] = axr_1.getValue(gcoord); data_2[7] = axr_2.getValue(gcoord); data_3[7] = axr_3.getValue(gcoord);//111
		j--; data_1[5] = axr_1.getValue(gcoord); data_2[5] = axr_2.getValue(gcoord); data_3[5] = axr_3.getValue(gcoord);//101
		k--; data_1[4] = axr_1.getValue(gcoord); data_2[4] = axr_2.getValue(gcoord); data_3[4] = axr_3.getValue(gcoord);//100
	}

	//trilinear interpolation with sse instructions
	//here weight is the x - floor(x), the distance to the lower end
	//the eight data is:
	//x lower: from low bit to high bit:
	//000
	//001
	//010
	//011
	//x higher: from low bit to high bit:
	//100
	//101
	//110
	//111
	static float trilinear_interpolate(const __m128& xlower, const __m128& xhigher, const __m128& weight) {
		//complement weight: 1.0 - weight
		__m128 cweight = _mm_sub_ps(_mm_set_ps1(1.0f), weight);

		//broadcase the original x weight and complement weight
		__m128 uniform_x_weight = _mm_shuffle_ps(weight, weight, 0b00000000);
		__m128 uniform_x_cweight = _mm_shuffle_ps(cweight, cweight, 0b00000000);

		//weight of the x direction taken into consideration
		__m128 newxlower = _mm_mul_ps(xlower, uniform_x_cweight);
		__m128 newxhigher = _mm_mul_ps(xhigher, uniform_x_weight);

		//results concentrated into four x lower points
		newxlower = _mm_add_ps(newxlower, newxhigher);

		//in xlower: high[y+z+, y+z-, y-z+, y-z-]low
		//lower half get complementary y weight, higher part get y weight
		__m128 y_weight = _mm_shuffle_ps(cweight, weight, 0b01010101);
		newxlower = _mm_mul_ps(newxlower, y_weight);

		//now z weight = h[wz, wz, cwz, cwz]l
		__m128 z_weight = _mm_shuffle_ps(cweight, weight, 0b10101010);

		//use integer shuffle to re-arrange the element inside the register
		//original order: 3:wz 2:wz 1:cwz 0:cwz
		//new order: 3:wz 1:cwz 2:wz 0:cwz
		__m128i z_weighti = _mm_castps_si128(z_weight);
		z_weighti = _mm_shuffle_epi32(z_weighti, 0b11011000);
		z_weight = _mm_castsi128_ps(z_weighti);
		//now z weight is h[wz cwz wz cwz]l
		newxlower = _mm_mul_ps(newxlower, z_weight);

		//reduce the elemet in zlower
		__m128 shuf = _mm_movehdup_ps(newxlower);        // broadcast elements 3,1 to 2,0
		__m128 sums = _mm_add_ps(newxlower, shuf);
		shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
		sums = _mm_add_ss(sums, shuf);
		return _mm_cvtss_f32(sums);
	}

	static void trilinear_interpolate2(float& out0, float& out1, const float* data0, const float* data1, const __m128& weight) {
		out0 = trilinear_interpolate(_mm_loadu_ps(data0), _mm_loadu_ps(data0 + 4), weight);
		out1 = trilinear_interpolate(_mm_loadu_ps(data1), _mm_loadu_ps(data1 + 4), weight);
	}

	static void trilinear_interpolate3(float& out0, float& out1, float& out2, const float* data0, const float* data1, const float* data2, const __m128& weight) {
		out0 = trilinear_interpolate(_mm_loadu_ps(data0), _mm_loadu_ps(data0 + 4), weight);
		out1 = trilinear_interpolate(_mm_loadu_ps(data1), _mm_loadu_ps(data1 + 4), weight);
		out2 = trilinear_interpolate(_mm_loadu_ps(data2), _mm_loadu_ps(data2 + 4), weight);
	}

	template<typename axr_t>
	static float sample(axr_t& axr, const __m128& ixyz) {
		__m128 floored_ixyz = _mm_round_ps(ixyz, _MM_FROUND_FLOOR);
		__m128i coord_ixyz = _mm_cvtps_epi32(floored_ixyz);
		int tempbase[4];
		_mm_storeu_ps((float*)tempbase, _mm_castsi128_ps(coord_ixyz));
		openvdb::Coord base(tempbase[0], tempbase[1], tempbase[2]);

		float data[8];
		get_eight_data(data, axr, base);

		__m128 weight = _mm_sub_ps(ixyz, floored_ixyz);
		return trilinear_interpolate(_mm_loadu_ps(data), _mm_loadu_ps(data + 4), weight);
	};

	template<typename axr_1_type, typename axr_2_type>
	static void sample2(float& out1, float& out2, axr_1_type& axr_1, axr_2_type& axr_2, const __m128& ixyz) {
		__m128 floored_ixyz = _mm_round_ps(ixyz, _MM_FROUND_FLOOR);
		__m128i coord_ixyz = _mm_cvtps_epi32(floored_ixyz);
		int tempbase[4];
		_mm_storeu_ps((float*)tempbase, _mm_castsi128_ps(coord_ixyz));
		openvdb::Coord base(tempbase[0], tempbase[1], tempbase[2]);
		float data1[8], data2[8];
		get_eight_data2(data1, data2, axr_1, axr_2, base);

		__m128 weight = _mm_sub_ps(ixyz, floored_ixyz);
		float tempweight[4];
		_mm_storeu_ps(tempweight, weight);
		trilinear_interpolate2(out1, out2, data1, data2, weight);
	};

	template<typename axr_1_type, typename axr_2_type, typename axr_3_type>
	static void sample3(float& out1, float& out2, float& out3, axr_1_type& axr_1, axr_2_type& axr_2, axr_3_type& axr_3, const __m128& ixyz) {
		__m128 floored_ixyz = _mm_round_ps(ixyz, _MM_FROUND_FLOOR);
		__m128i coord_ixyz = _mm_cvtps_epi32(floored_ixyz);
		int tempbase[4];
		_mm_storeu_ps((float*)tempbase, _mm_castsi128_ps(coord_ixyz));
		openvdb::Coord base(tempbase[0], tempbase[1], tempbase[2]);
		float data1[8], data2[8], data3[8];
		get_eight_data3(data1, data2, data3, axr_1, axr_2, axr_3, base);

		__m128 weight = _mm_sub_ps(ixyz, floored_ixyz);
		trilinear_interpolate3(out1, out2, out3, data1, data2, data3, weight);
	};
};

struct StaggeredBoxSampler {
	template<typename axr_type>
	static __m128 sample(axr_type* axr, const __m128& xyz) {
		__m128 shifted_val = xyz;
		shifted_val = _mm_add_ps(shifted_val, _mm_set_ps1(0.5f));
		float tempv[4] = { 0.f,0.f,0.f,0.f };
		tempv[0] = BoxSampler::sample(axr[0], _mm_blend_ps(xyz, shifted_val, 0b0001));
		tempv[1] = BoxSampler::sample(axr[1], _mm_blend_ps(xyz, shifted_val, 0b0010));
		tempv[2] = BoxSampler::sample(axr[2], _mm_blend_ps(xyz, shifted_val, 0b0100));

		return _mm_loadu_ps(tempv);
	};

	template<typename axr_1_type, typename axr_2_type>
	static void sample2(__m128& v1, __m128& v2, axr_1_type* axr_1, axr_2_type* axr_2, const __m128& xyz) {
		__m128 shifted_val = xyz;
		shifted_val = _mm_add_ps(shifted_val, _mm_set_ps1(0.5f));
		float tempv1[4] = { 0.f,0.f,0.f,0.f };
		float tempv2[4] = { 0.f,0.f,0.f,0.f };
		BoxSampler::sample2(tempv1[0], tempv2[0], axr_1[0], axr_2[0], _mm_blend_ps(xyz, shifted_val, 0b0001));
		BoxSampler::sample2(tempv1[1], tempv2[1], axr_1[1], axr_2[1], _mm_blend_ps(xyz, shifted_val, 0b0010));
		BoxSampler::sample2(tempv1[2], tempv2[2], axr_1[2], axr_2[2], _mm_blend_ps(xyz, shifted_val, 0b0100));
		v1 = _mm_loadu_ps(tempv1);
		v2 = _mm_loadu_ps(tempv2);
	};

	template<typename axr_1_type, typename axr_2_type, typename axr_3_type>
	static void sample3(__m128& v1, __m128& v2, __m128& v3, axr_1_type* axr_1, axr_2_type* axr_2, axr_3_type* axr_3, const __m128& xyz) {
		__m128 shifted_val = xyz;
		shifted_val = _mm_add_ps(shifted_val, _mm_set_ps1(0.5f));
		float tempv1[4] = { 0.f,0.f,0.f,0.f };
		float tempv2[4] = { 0.f,0.f,0.f,0.f };
		float tempv3[4] = { 0.f,0.f,0.f,0.f };
		BoxSampler::sample3(tempv1[0], tempv2[0], tempv3[0], axr_1[0], axr_2[0], axr_3[0], _mm_blend_ps(xyz, shifted_val, 0b0001));
		BoxSampler::sample3(tempv1[1], tempv2[1], tempv3[1], axr_1[1], axr_2[1], axr_3[1], _mm_blend_ps(xyz, shifted_val, 0b0010));
		BoxSampler::sample3(tempv1[2], tempv2[2], tempv3[2], axr_1[2], axr_2[2], axr_3[2], _mm_blend_ps(xyz, shifted_val, 0b0100));
		v1 = _mm_loadu_ps(tempv1);
		v2 = _mm_loadu_ps(tempv2);
		v3 = _mm_loadu_ps(tempv3);
	};

};
}

namespace {
//first step, for some of the velocity voxel
//one of its weights is non zero because some particle touches it
//however the velocity of other velocity is missing
//deduce it from its 27 neighboring voxel that has non-zero weights
//on this component
struct normalize_p2g_velocity {
	normalize_p2g_velocity(
		openvdb::Vec3fGrid::Ptr in_velocity_weights,
		packed_FloatGrid3 in_out_packed_velocity
	) : original_weights_accessor(in_velocity_weights->getConstUnsafeAccessor()) {
		m_packed_velocity = in_out_packed_velocity;
	}
	normalize_p2g_velocity(const normalize_p2g_velocity& other) :
		original_weights_accessor(other.original_weights_accessor),
		m_packed_velocity(other.m_packed_velocity)
	{
		original_weights_accessor.clear();
	}

	void operator()(openvdb::Vec3fGrid::TreeType::LeafNodeType& vel_leaf, openvdb::Index leafpos) const {
		const auto* oweight_leaf = original_weights_accessor.probeConstLeaf(vel_leaf.origin());
		float epsl = 0.f;
		openvdb::FloatTree::LeafNodeType* component_leaf[3];

		for (int i = 0; i < 3; i++) {
			component_leaf[i] = m_packed_velocity.v[i]->tree().probeLeaf(vel_leaf.origin());
		}

		for (int i_component = 0; i_component < 3; i_component++) {
			for (auto iter = vel_leaf.beginValueOn(); iter; ++iter) {

				openvdb::Vec3f vel = vel_leaf.getValue(iter.offset());
				//check its three component
				float weight = oweight_leaf->getValue(iter.offset())[i_component];

				//missing velocities are strictly zero
				if (weight == 0) {
					component_leaf[i_component]->setValueOff(iter.offset(), 0.f);
				}
				else {
					vel[i_component] /= weight + epsl;
					component_leaf[i_component]->setValueOn(iter.offset(), vel[i_component]);
				}
			}//end for all voxels in this leaf node
		}//end for three component find missing velocity
	}//end operator()

	openvdb::Vec3fGrid::ConstUnsafeAccessor original_weights_accessor;

	packed_FloatGrid3 m_packed_velocity;
};
}

namespace {
struct custom_integrator {
	custom_integrator(packed_FloatGrid3 in_velocity, float in_dx, float in_dt) :
		vaxr{ in_velocity.v[0]->tree(), in_velocity.v[1]->tree(), in_velocity.v[2]->tree() } {
		dx = in_dx;
		invdx = 1.0f / dx;
		dt = in_dt;
		dtinvx = dt / in_dx;
		packed_dtinvx = _mm_set_ps1(dtinvx);
		packed_half_dtinvx = _mm_set_ps1(dtinvx * 0.5f);
		packed_sixth_dtinvx = _mm_set_ps1(dtinvx * 1.0f / 6.0f);
	}

	custom_integrator(const custom_integrator& other) :
		vaxr{ other.vaxr[0], other.vaxr[1], other.vaxr[2] },
		dx(other.dx), invdx(other.invdx), dt(other.dt),
		dtinvx(other.dtinvx),
		packed_dtinvx(other.packed_dtinvx),
		packed_half_dtinvx(other.packed_half_dtinvx),
		packed_sixth_dtinvx(other.packed_sixth_dtinvx){
		for (int i = 0; i < 3; i++) {
			vaxr[i].clear();
		}
	}

	void packed_integrate1(__m128& ipos, const __m128& V0) const {
		ipos = _mm_add_ps(ipos, _mm_mul_ps(packed_dtinvx, V0));
	}

	void packed_integrate2(__m128& ipos, const __m128& V0) const {
		__m128 midpoint = _mm_add_ps(ipos, _mm_mul_ps(packed_half_dtinvx, V0));
		__m128 midv = StaggeredBoxSampler::sample(vaxr, midpoint);
		ipos = _mm_add_ps(ipos, _mm_mul_ps(midv, packed_dtinvx));
	}

	void packed_integrate3(__m128& ipos, const __m128& V0) const {
		/*openvdb::Vec3f V1 = StaggeredBoxSampler::sample(vaxr, ipos + 0.5f * V0 * dtinvx);
		openvdb::Vec3f V2 = StaggeredBoxSampler::sample(vaxr, ipos + dtinvx * (2.0f * V1 - V0));
		ipos += dtinvx * (V0 + 4.0f * V1 + V2) * (1.0f / 6.0f);*/
		__m128 X1 = _mm_add_ps(ipos, _mm_mul_ps(packed_half_dtinvx, V0));
		__m128 V1 = StaggeredBoxSampler::sample(vaxr, X1);
		__m128 X2 = _mm_add_ps(ipos, _mm_mul_ps(packed_dtinvx, _mm_sub_ps(_mm_add_ps(V1, V1), V0)));
		__m128 V2 = StaggeredBoxSampler::sample(vaxr, X2);
		ipos = _mm_add_ps(ipos, _mm_mul_ps(packed_sixth_dtinvx, _mm_add_ps(_mm_add_ps(V0, V2), _mm_mul_ps(V1, _mm_set_ps1(4.0f)))));
	}

	void packed_integrate4(__m128& ipos, const __m128& V0) const {
		/*openvdb::Vec3f V1 = StaggeredBoxSampler::sample(vaxr, ipos + 0.5f * V0 * dtinvx);
		openvdb::Vec3f V2 = StaggeredBoxSampler::sample(vaxr, ipos + 0.5f * V1 * dtinvx);
		openvdb::Vec3f V3 = StaggeredBoxSampler::sample(vaxr, ipos + V2 * dtinvx);
		ipos += dtinvx * (V0 + 2.0f * (V1 + V2) + V3) * (1.0f / 6.0f);*/
		__m128 X1 = _mm_add_ps(ipos, _mm_mul_ps(packed_half_dtinvx, V0));
		__m128 V1 = StaggeredBoxSampler::sample(vaxr, X1);
		__m128 X2 = _mm_add_ps(ipos, _mm_mul_ps(packed_half_dtinvx, V1));
		__m128 V2 = StaggeredBoxSampler::sample(vaxr, X2);
		__m128 X3 = _mm_add_ps(ipos, _mm_mul_ps(packed_dtinvx, V2));
		__m128 V3 = StaggeredBoxSampler::sample(vaxr, X3);
		__m128 sum_V1_V2 = _mm_add_ps(V1, V2);
		ipos = _mm_add_ps(ipos, _mm_mul_ps(packed_sixth_dtinvx, _mm_add_ps(_mm_add_ps(V0, V3), _mm_add_ps(sum_V1_V2, sum_V1_V2))));
	}

	float dx;
	float invdx;
	float dtinvx;
	float dt;
	__m128 packed_dtinvx, packed_half_dtinvx, packed_sixth_dtinvx;
	mutable openvdb::FloatGrid::ConstUnsafeAccessor vaxr[3];
};

struct point_to_counter_reducer {

	struct to_oi_olp {
		to_oi_olp(uint16_t to, openvdb::Index32 oi, openvdb::Index32 olp) {
			target_offset = to;
			original_index = oi;
			original_leaf_pos = olp;
		}

		to_oi_olp() {}

		openvdb::Index32 original_index;
		openvdb::Index32 original_leaf_pos;
		uint16_t target_offset;
	};

	using to_oi_olp_container_type = custom_vector_type<to_oi_olp>;

	point_to_counter_reducer(
		const float in_dt,
		const float in_dx,
		packed_FloatGrid3 in_velocity,
		packed_FloatGrid3 in_velocity_to_be_advected,
		packed_FloatGrid3 in_old_velocity,
		const openvdb::FloatGrid& in_solid_sdf,
		const openvdb::Vec3fGrid& in_center_solid_grad,
		const openvdb::FloatGrid& in_center_solid_vel_n,
		float pic_component,
		const std::vector<openvdb::points::PointDataTree::LeafNodeType*>& in_particles,
		const openvdb::points::PointDataGrid& in_particles_grid,
		int RK_order) : dt(in_dt),
		m_rk_order(RK_order), m_dx(in_dx), m_invdx(1.0f / in_dx),
		m_velocity(in_velocity),
		m_velocity_to_be_advected(in_velocity_to_be_advected),
		m_old_velocity(in_old_velocity),
		m_solid_sdf(in_solid_sdf),
		m_center_solidgrad(in_center_solid_grad),
		m_center_solidveln(in_center_solid_vel_n),
		m_pic_component(pic_component),
		m_particles(in_particles),
		m_particles_grid(in_particles_grid)
	{
		m_integrator = std::make_unique<custom_integrator>(in_velocity, in_dx, in_dt);
		//create the interior voxel indicator so surface particles are advected with only forward euler
		auto air_mask = openvdb::BoolGrid::create();
		air_mask->setTree(std::make_shared<openvdb::BoolTree>(m_particles_grid.tree(), /*bgval*/ false, openvdb::TopologyCopy()));
		openvdb::tools::dilateActiveValues(air_mask->tree(), 1, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);
		air_mask->tree().topologyDifference(m_particles_grid.tree());
		openvdb::tools::dilateActiveValues(air_mask->tree(), 1, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);
		//at this point air_mask consists of 1 layer of air particles, 1 layer of near air, and 1 layer of extra air
		m_surface_particle_indicator = air_mask;
	}

	point_to_counter_reducer(const point_to_counter_reducer& other, tbb::split) :
		dt(other.dt), m_rk_order(other.m_rk_order), m_dx(other.m_dx), m_invdx(other.m_invdx),
		m_velocity(other.m_velocity),
		m_velocity_to_be_advected(other.m_velocity_to_be_advected),
		m_old_velocity(other.m_old_velocity),
		m_pic_component(other.m_pic_component),
		m_solid_sdf(other.m_solid_sdf),
		m_center_solidgrad(other.m_center_solidgrad),
		m_center_solidveln(other.m_center_solidveln),
		m_particles(other.m_particles),
		m_particles_grid(other.m_particles_grid),
		m_surface_particle_indicator(other.m_surface_particle_indicator) {
		m_integrator = std::make_unique<custom_integrator>(m_velocity, m_dx, dt);
	}


	//loop over ranges of flattened particle leaves
	void operator()(const tbb::blocked_range<openvdb::Index>& r) {
		using  namespace openvdb::tools::local_util;

		openvdb::FloatGrid::ConstUnsafeAccessor vaxr[3] = {
			m_velocity.v[0]->tree(),
			m_velocity.v[1]->tree(),
			m_velocity.v[2]->tree() };
		openvdb::FloatGrid::ConstUnsafeAccessor v_tobe_adv_axr[3] = {
			m_velocity_to_be_advected.v[0]->tree(),
			m_velocity_to_be_advected.v[1]->tree(),
			m_velocity_to_be_advected.v[2]->tree() };
		openvdb::FloatGrid::ConstUnsafeAccessor old_vaxr[3] = {
			m_old_velocity.v[0]->tree(),
			m_old_velocity.v[1]->tree(),
			m_old_velocity.v[2]->tree() };

		auto surface_indicator_axr{ m_surface_particle_indicator->getConstUnsafeAccessor() };
		auto sdfaxr{ m_solid_sdf.getConstUnsafeAccessor() };
		auto naxr{ m_center_solidgrad.getConstUnsafeAccessor() };
		auto vnaxr{ m_center_solidveln.getConstUnsafeAccessor() };

		//leaf iter
		for (auto liter = r.begin(); liter != r.end(); ++liter) {
			auto& leaf = *m_particles[liter];
			auto surface_indicator_leaf = surface_indicator_axr.probeConstLeaf(leaf.origin());

			//attributes
			// Attribute reader
			// Extract the position attribute from the leaf by name (P is position).
			openvdb::points::AttributeArray& positionArray =
				leaf.attributeArray("P");
			// Extract the velocity attribute from the leaf by name (v is velocity).
			openvdb::points::AttributeArray& velocityArray =
				leaf.attributeArray("v");

			// Create read handles for position and velocity
			openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_Object_OpenVDB::PositionCodec> positionHandle(positionArray);
			openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_Object_OpenVDB::VelocityCodec> velocityHandle(velocityArray);

			to_oi_olp_container_type* writing_offset_index_leafpos;
			//old leaf beging and end
			openvdb::Coord olbegin{ openvdb::Coord::max() };
			openvdb::Coord olend{ openvdb::Coord::min() };
			float flip_component = (1.0f - m_pic_component);
			float deep_threshold = -2.0f * m_dx;
			float touch_threshold = 0.2f * m_dx;
			float invdx = 1.0f / m_dx;
			float dtinvdx = dt / m_dx;

			bool adv_same_field = m_velocity.v[0]->treePtr() == m_velocity_to_be_advected.v[0]->treePtr();

			__m128 packed_flip_component = _mm_set_ps1(flip_component);
			__m128 packed_dtinvdx = _mm_set_ps1(dtinvdx);
			__m128 packed_half = _mm_set_ps1(0.5f);
			for (auto viter = leaf.beginValueOn(); viter; ++viter) {
				auto idxend = viter.getValue();
				openvdb::PointDataIndex32 idxbegin = 0;
				if (viter.offset() != 0) {
					idxbegin = leaf.getValue(viter.offset() - 1);
				}

				auto pIvpos = viter.getCoord();
				__m128 packed_pIvpos = _mm_cvtepi32_ps(_mm_set_epi32(0, pIvpos.z(), pIvpos.y(), pIvpos.x()));

				for (auto pidx = idxbegin; pidx < idxend; pidx = pidx + 1) {
					auto voxelpos = positionHandle.get(pidx);
					float tempvoxelpos[4] = { voxelpos.x(),voxelpos.y(),voxelpos.z(), 0.f };
					__m128 packed_pIspos = _mm_add_ps(packed_pIvpos, _mm_loadu_ps(tempvoxelpos));

					openvdb::Vec3f particle_vel = velocityHandle.get(pidx);
					float tempparticle_vel[4] = { particle_vel.x(),particle_vel.y(),particle_vel.z(), 0.f };
					__m128 packed_particle_vel = _mm_loadu_ps(tempparticle_vel);

					__m128 packed_adv_vel, packed_old_vel, packed_carried_vel;
					if (adv_same_field) {
						StaggeredBoxSampler::sample2(packed_adv_vel, packed_old_vel, vaxr, old_vaxr, packed_pIspos);
						packed_carried_vel = packed_adv_vel;
					}
					else {
						StaggeredBoxSampler::sample3(packed_adv_vel, packed_old_vel, packed_carried_vel, vaxr, old_vaxr, v_tobe_adv_axr, packed_pIspos);
					}

					//update the velocity of the particle
					/*particle_vel = (m_pic_component)*adv_vel + (1.0f - m_pic_component) * (adv_vel - old_vel + particle_vel);*/
					//particle_vel = carried_vel + flip_component * (-old_vel + particle_vel);
					packed_particle_vel = _mm_add_ps(
						packed_carried_vel, _mm_mul_ps(
							packed_flip_component, _mm_sub_ps(
								packed_particle_vel, packed_old_vel)));

					_mm_storeu_ps(tempparticle_vel, packed_particle_vel);
					particle_vel = openvdb::Vec3f(tempparticle_vel);


					__m128 packed_pItpos = packed_pIspos;
					//for surface particles, only its current velocity are reliable
					//so use this velocity for advection
					//for interior particles, they may enjoy higher order advection scheme.
					if ((m_rk_order > 1) && !surface_indicator_leaf->isValueOn(viter.offset())) {
						//if (m_rk_order > 1) {
						switch (m_rk_order) {
						default:
						case 2:
							m_integrator->packed_integrate2(packed_pItpos, packed_adv_vel);
							break;
						case 3:
							m_integrator->packed_integrate3(packed_pItpos, packed_adv_vel);
							break;
						case 4:
							m_integrator->packed_integrate4(packed_pItpos, packed_adv_vel);
							break;
						}
					}
					else {
						//pItpos = pIspos + adv_vel * dtinvdx;
						packed_pItpos = _mm_add_ps(packed_pItpos, _mm_mul_ps(packed_adv_vel, packed_dtinvdx));
					}
					//shifted particle index position
					//auto spitpos = pItpos + openvdb::Vec3f{ 0.5f };
					__m128 floored_itpos = _mm_round_ps(_mm_add_ps(packed_pItpos, packed_half), _MM_FROUND_FLOOR);
					__m128i coord_itpos = _mm_cvtps_epi32(floored_itpos);
					int temptcoord[4];
					_mm_storeu_ps((float*)temptcoord, _mm_castsi128_ps(coord_itpos));
					openvdb::Coord ptCoord = openvdb::Coord(temptcoord);

					//collision detection, see if this coordinate is inside the solid
					if (naxr.isValueOn(ptCoord)) {
						float p_solid_ipos[4];
						_mm_storeu_ps(p_solid_ipos, _mm_add_ps(packed_pItpos, packed_half));
						float new_pos_solid_sdf = openvdb::tools::BoxSampler::sample(sdfaxr, openvdb::Vec3R(p_solid_ipos));
						new_pos_solid_sdf -= touch_threshold;
						if (new_pos_solid_sdf < 0) {
							if (new_pos_solid_sdf < deep_threshold) {
								//too deep, just continue to next particle
								continue;
							}
							float temp_pipos[4];
							_mm_storeu_ps(temp_pipos, packed_pItpos);
							//inside, but still can be saved, use the surface normal to move back the particle
							auto snormal = openvdb::tools::BoxSampler::sample(naxr, openvdb::Vec3R(temp_pipos));

							__m128 packed_snormal = _mm_set_ps(0.f, snormal.z(), snormal.y(), snormal.x());

							//handle velocity bounces
							particle_vel += (vnaxr.getValue(ptCoord) - snormal.dot(particle_vel)) * snormal;
							//move particle out
							//pItpos -= new_pos_solid_sdf * snormal * invdx ;
							packed_pItpos = _mm_sub_ps(packed_pItpos, _mm_mul_ps(packed_snormal, _mm_set_ps1(new_pos_solid_sdf * invdx)));

							floored_itpos = _mm_round_ps(_mm_add_ps(packed_pItpos, packed_half), _MM_FROUND_FLOOR);
							coord_itpos = _mm_cvtps_epi32(floored_itpos);
							_mm_storeu_ps((float*)temptcoord, _mm_castsi128_ps(coord_itpos));
							ptCoord = openvdb::Coord(temptcoord);
						}//end if particle is truly inside solid

					}//end if surface normal exist

					//directly change the original attribute to the target voxel position
					//later it will be transfered to the new position
					__m128 packed_pvoxelpos = _mm_sub_ps(packed_pItpos, floored_itpos);

					_mm_storeu_ps(tempvoxelpos, packed_pvoxelpos);

					positionHandle.set(pidx, openvdb::Vec3f(tempvoxelpos));
					velocityHandle.set(pidx, particle_vel);
					//check if we are writing to the previous leaf?

					const openvdb::Coord torigin = ptCoord & (~7);
					if (torigin == olbegin) {
						//increment the counter
						uint16_t toffset = leaf.coordToOffset(ptCoord);
						//append the velocity and index space position
						writing_offset_index_leafpos->push_back(to_oi_olp{ toffset, pidx, liter });
					}//end if writing to same leaf
					else {
						olbegin = torigin;
						auto requested_iter = toffset_oindex_oleafpos_hashmap.find(torigin);
						if (requested_iter == toffset_oindex_oleafpos_hashmap.end()) {
							std::unique_ptr<to_oi_olp_container_type> new_tuplevecptr = std::make_unique<to_oi_olp_container_type>();
							writing_offset_index_leafpos = new_tuplevecptr.get();
							toffset_oindex_oleafpos_hashmap[torigin] = std::move(new_tuplevecptr);
						}
						else {
							writing_offset_index_leafpos = requested_iter->second.get();
						}

						uint16_t toffset = leaf.coordToOffset(ptCoord);
						//set the bounding box
						writing_offset_index_leafpos->push_back(to_oi_olp{ toffset, pidx, liter });
					}//end else writing to the same leaf
				}//end loop over all particles
			}//end voxel value loop
		}//end for range leaves
	}//end operator

	void join(point_to_counter_reducer& other) {
		for (auto other_tuplevec = other.toffset_oindex_oleafpos_hashmap.begin();
			other_tuplevec != other.toffset_oindex_oleafpos_hashmap.end(); ++other_tuplevec) {
			auto itr_in_this = toffset_oindex_oleafpos_hashmap.find(other_tuplevec->first);
			if (itr_in_this != toffset_oindex_oleafpos_hashmap.end()) {
				auto original_size = itr_in_this->second->size();
				itr_in_this->second->resize(original_size + other_tuplevec->second->size());
				std::copy(other_tuplevec->second->begin(), other_tuplevec->second->end(), itr_in_this->second->begin() + original_size);
			}
			else {
				toffset_oindex_oleafpos_hashmap[other_tuplevec->first] = std::move(other_tuplevec->second);
			}
		}
	}


	//velocity integrator
	std::unique_ptr<custom_integrator> m_integrator;

	//time step
	const float dt;

	const int m_rk_order;
	//index to world transform
	//for the particles as well as the velocity
	const float m_dx;
	const float m_invdx;

	//the velocity field used to advect the particles
	packed_FloatGrid3 m_velocity;
	packed_FloatGrid3 m_velocity_to_be_advected;
	packed_FloatGrid3 m_old_velocity;
	float m_pic_component;

	const openvdb::FloatGrid& m_solid_sdf;
	const openvdb::Vec3fGrid& m_center_solidgrad;
	const openvdb::FloatGrid& m_center_solidveln;

	//the source particles
	const std::vector<openvdb::points::PointDataTree::LeafNodeType*>& m_particles;

	//particles grid are used to generate the mask
	const openvdb::points::PointDataGrid& m_particles_grid;
	openvdb::BoolGrid::Ptr m_surface_particle_indicator;

	//hashmap storing target voxel offset, original attribute index, original leaf position
	std::unordered_map<
		openvdb::Coord, std::unique_ptr<to_oi_olp_container_type>> toffset_oindex_oleafpos_hashmap;
};//end reducer

}//end namespace

namespace {
struct p2g_collector {
	p2g_collector(openvdb::FloatGrid::Ptr in_liquid_sdf,
		openvdb::Vec3fGrid::Ptr in_velocity,
		openvdb::Vec3fGrid::Ptr in_velocity_weight,
		openvdb::points::PointDataGrid::Ptr in_particles,
		float in_particle_radius) {
		m_liquid_sdf = in_liquid_sdf;
		m_velocity = in_velocity;
		m_velocity_weight = in_velocity_weight;
		m_particles = in_particles;
		m_particle_radius = in_particle_radius;
		//set the loop order and offset of the u v w phi sampling point
		//with respect to the center voxel center

		//u:  (-0.5, 0, 0)
		//v:  (0, -0.5, 0)
		//w:  (0, 0, -0.5)
		//phi:(0, 0, 0)
		//note we use set so the 
		//R0 R1 R2 R3 correspond to the 0,1,2,3 argument
		//when later extracting using _mm_storer_ps, we get float[0] = arg0 correctly
		//see http://wwwuser.gwdg.de/~parallel/intel_compiler_doc_91/main_cls/mergedProjects/intref_cls/common/intref_sse_store.htm
		//see http://wwwuser.gwdg.de/~parallel/intel_compiler_doc_91/main_cls/mergedProjects/intref_cls/common/intref_sse_set.htm

		__m128 xpack = _mm_set_ps(-0.5f, 0.f, 0.f, 0.f);
		__m128 ypack = _mm_set_ps(0.f, -0.5f, 0.f, 0.f);
		__m128 zpack = _mm_set_ps(0.f, 0.f, -0.5f, 0.f);
		for (int ivoxel = 0; ivoxel < 27; ivoxel++) {
			int ijk = ivoxel;
			int basex = ijk / 9; ijk -= 9 * basex;
			int basey = ijk / 3; ijk -= 3 * basey;
			int basez = ijk;
			//becomes -1 -> 1

			basex -= 1; basey -= 1; basez -= 1;
			//broadcast four float as the base
			__m128 basex4 = _mm_set_ps1(float(basex));
			__m128 basey4 = _mm_set_ps1(float(basey));
			__m128 basez4 = _mm_set_ps1(float(basez));

			loop_order.at(ivoxel) = openvdb::Coord{ basex, basey, basez };

			x_offset_to_center_voxel_center.at(ivoxel) = _mm_add_ps(basex4, xpack);
			y_offset_to_center_voxel_center.at(ivoxel) = _mm_add_ps(basey4, ypack);
			z_offset_to_center_voxel_center.at(ivoxel) = _mm_add_ps(basez4, zpack);
		}
	}


	//packed structure storing a particle leaf
	//its attribute handle
	struct particle_leaf {
		particle_leaf() {
			m_leafptr = nullptr;
			m_p_handle_ptr.reset();
			m_v_handle_ptr.reset();
		}

		//iterator over points in a voxel
		struct particle_iter {
			particle_iter() :m_parent(nullptr) {
				m_item = 0;
				m_indexend = 0;
				m_indexbegin = 0;
			}

			particle_iter(const particle_leaf* in_parent, openvdb::Index in_offset) {
				set(in_parent, in_offset);
			}

			particle_iter(const particle_iter& other) = default;

			void set(const particle_leaf* in_parent, openvdb::Index in_offset) {
				m_parent = in_parent;

				if (!in_parent->m_leafptr) {
					m_item = 0;
					m_indexbegin = 0;
					m_indexend = 0;
					return;
				}
				m_indexend = in_parent->m_leafptr->getValue(in_offset);
				if (in_offset == 0) {
					m_indexbegin = 0;
				}
				else {
					m_indexbegin = in_parent->m_leafptr->getValue(in_offset - 1);
				}
				m_item = m_indexbegin;
			}

			void operator =(const particle_iter& other) {
				m_item = other.m_item;
				m_parent = other.m_parent;
				m_indexbegin = other.m_indexbegin;
				m_indexend = other.m_indexend;
			}

			operator bool() const {
				return m_item < m_indexend;
			}

			particle_iter& operator ++() {
				m_item++;
				return *this;
			}

			openvdb::Index operator*()  const {
				return m_item;
			}

			openvdb::Index m_item;
			openvdb::Index m_indexbegin, m_indexend;
			const particle_leaf* m_parent;
		};


		particle_leaf(const openvdb::points::PointDataTree::LeafNodeType* in_leaf_ptr) : m_leafptr(in_leaf_ptr) {
			if (in_leaf_ptr) {
				// Attribute reader
				// Extract the position attribute from the leaf by name (P is position).
				const openvdb::points::AttributeArray& positionArray =
					in_leaf_ptr->attributeArray("P");
				// Extract the velocity attribute from the leaf by name (v is velocity).
				const openvdb::points::AttributeArray& velocityArray =
					in_leaf_ptr->attributeArray("v");

				//// Create read handles for position and velocity
				m_p_handle_ptr = openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_Object_OpenVDB::PositionCodec>::create(positionArray);
				m_v_handle_ptr = openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_Object_OpenVDB::VelocityCodec>::create(velocityArray);
			}
		}

		//default?
		particle_leaf(const particle_leaf& other) {
			m_leafptr = other.m_leafptr;
			m_p_handle_ptr = other.m_p_handle_ptr;
			m_v_handle_ptr = other.m_v_handle_ptr;
		}

		particle_iter get_particle_iter(openvdb::Index offset) const {
			return particle_iter(this, offset);
		}

		particle_iter get_particle_iter(const openvdb::Coord& lgxyz) const {
			return particle_iter(this, m_leafptr->coordToOffset(lgxyz));
		}

		operator bool() const {
			return m_leafptr != nullptr;
		}

		openvdb::Vec3f get_p(openvdb::Index pos) const {
			return m_p_handle_ptr->get(pos);
		}

		openvdb::Vec3f get_v(openvdb::Index pos) const {
			return m_v_handle_ptr->get(pos);
		}

		openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_Object_OpenVDB::PositionCodec>::Ptr m_p_handle_ptr;
		openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_Object_OpenVDB::VelocityCodec>::Ptr m_v_handle_ptr;
		const openvdb::points::PointDataTree::LeafNodeType* m_leafptr;
	};

	void fill_particle_leafs(std::array<particle_leaf, 27>& in_out_leaves, const openvdb::Coord& center_origin, const openvdb::points::PointDataTree& in_particle_tree) const {
		//for all its neighbor nodes
		int counter = 0;
		for (int ii = -8; ii <= 8; ii += 8) {
			for (int jj = -8; jj <= 8; jj += 8) {
				for (int kk = -8; kk <= 8; kk += 8) {
					auto leafptr = in_particle_tree.probeConstLeaf(center_origin.offsetBy(ii, jj, kk));
					in_out_leaves[counter++] = particle_leaf(leafptr);
				}
			}
		}
	}

	//given the 27 particle leaves, 
	//iterate all particles that could possibly contribute to the center leaf
	//each iterator can return the position, velocity, local coordinate relative
	//to the center leaf
	struct all_particle_iterator {
		all_particle_iterator(const std::array<particle_leaf, 27>& in_leaves) :
			m_leaves(in_leaves) {
			m_leafpos = 0;
			at_voxel = 0;
			//initially on the corner
			m_leafxyz = { -8,-8,-8 };
			//it runs from [{-1,-1,-1}, {9,9,9} )
			//the size is 1+8+1 = 10
			//so there are at most 1000 voxels to iterate
			m_center_xyz = openvdb::Coord{ -1,-1,-1 };
			auto offset = 511u;
			//printf("setting voxel iter at offset%d\n",offset);
			m_voxel_particle_iter.set(&m_leaves[m_leafpos], offset);
			//printf("voxel_iter set\n");
			if (!m_voxel_particle_iter) {
				move_to_next_on_voxel();
			}
		}

		void move_to_next_on_voxel() {
			do {
				//ran out of attribute in this voxel
				//find the next on voxel that will contribute
				//to the center voxel
				//allowed range: [-1,8] in x,y,z
				m_center_xyz[2]++;

				//only when it freshly turns 0 and 8 it cross the leaf border
				if (m_center_xyz[2] == 0) { m_leafpos++; }
				else if (m_center_xyz[2] == 8) { m_leafpos++; }
				else
					if (m_center_xyz[2] == 9) {
						m_center_xyz[2] = -1; m_leafpos -= 2;

						m_center_xyz[1]++;
						if (m_center_xyz[1] == 0) { m_leafpos += 3; }
						else if (m_center_xyz[1] == 8) { m_leafpos += 3; }
						else
							if (m_center_xyz[1] == 9) {
								m_center_xyz[1] = -1; m_leafpos -= 6;

								m_center_xyz[0]++;
								if (m_center_xyz[0] == 0) { m_leafpos += 9; }
								else if (m_center_xyz[0] == 8) { m_leafpos += 9; }
								else
									if (m_center_xyz[0] == 9) {
										m_leafpos += 9;
										if (m_leafpos != 27) {
											printf("particle iterator increment error! actual leafpos:%d\n", m_leafpos);
											exit(-1);
										}
									};
							}//end if y overflow
					}//end if z overflow

					//check leaf bount
				if (m_leafpos >= 27) return;

				//this leaf is empty
				if (m_leaves[m_leafpos].m_leafptr == nullptr) { continue; };

				int offset = m_leaves[m_leafpos].m_leafptr->coordToOffset(m_center_xyz);
				if (m_leaves[m_leafpos].m_leafptr->isValueOn(offset)) {
					m_voxel_particle_iter.set(&m_leaves[m_leafpos], offset);
					auto itercoord = m_center_xyz;
					return;
				}
			} while (true);
		}//end move to next on voxel



		openvdb::Coord getCoord() const {
			return m_center_xyz;
		}

		openvdb::Vec3f getP() const {
			return m_leaves[m_leafpos].get_p(*m_voxel_particle_iter);
		}

		openvdb::Vec3f getv() const {
			return m_leaves[m_leafpos].get_v(*m_voxel_particle_iter);
		}

		all_particle_iterator& operator++() {
			++m_voxel_particle_iter;
			//printf("advancing index %d at voxel %d\n", *m_voxel_particle_iter, at_voxel);
			if (!m_voxel_particle_iter) {
				move_to_next_on_voxel();
			}//end if ran out of particles in this voxel
			return *this;
		}

		operator bool() const {
			return m_voxel_particle_iter;
		}

		//local coordinates relative to the center leaf
		openvdb::Coord m_center_xyz;

		openvdb::Coord m_g_xyz;
		int at_voxel;
		//attribute index in corresponding 
		particle_leaf::particle_iter m_voxel_particle_iter;

		//0-26, the position in the particle leaf
		uint32_t m_leafpos;
		openvdb::Coord m_leafxyz;

		const std::array<particle_leaf, 27>& m_leaves;
	};

	//to be use by the velocity grid manager
	//collect the contribution of particles in current leaf
	//and its neighbor leafs
	void operator() (openvdb::Vec3fTree::LeafNodeType& leaf, openvdb::Index leafpos) const {
		auto new_vel_leaf = m_velocity->tree().probeLeaf(leaf.origin());
		auto new_vel_weights_leaf = m_velocity_weight->tree().probeLeaf(leaf.origin());
		auto new_sdf_leaf = m_liquid_sdf->tree().probeLeaf(leaf.origin());

		const __m128 absmask = _mm_castsi128_ps(_mm_set1_epi32(~(1 << 31)));
		const __m128 float1x4 = _mm_set_ps1(float(1));
		const __m128 float0x4 = _mm_set_ps1(float(0));

		//scatter style transfer
		//loop over all particles
		float tophix, tophiy, tophiz;
		float dx = m_liquid_sdf->voxelSize()[0];

		std::array<particle_leaf, 27> particle_leaves;
		fill_particle_leafs(particle_leaves, leaf.origin(), m_particles->tree());

		auto iter{ all_particle_iterator(particle_leaves) };
		//pre-allocated space for computation
		__m128 x_dist_particle_to_sample;
		__m128 y_dist_particle_to_sample;
		__m128 z_dist_particle_to_sample;
		__m128 comp_result;
		__m128 comp_target;
		//the distance to the other cell center will always be greater
		//than zero, hence the last predicate is always true
		comp_target = _mm_set_ps(1.f, 1.f, 1.f, -1.f);
		float dist_to_phi_sample;
		//particle iterator
		for (; iter; ++iter) {

			auto voxelpos = iter.getP();
			auto pvel = iter.getv();
			//broadcast the variables
			__m128 particle_x = _mm_set_ps1(voxelpos[0]);
			__m128 particle_y = _mm_set_ps1(voxelpos[1]);
			__m128 particle_z = _mm_set_ps1(voxelpos[2]);

			//calculate the distance to the 27 neib uvw phi samples
			for (int ivoxel = 0; ivoxel < 27; ivoxel++) {

				//is it writing to this leaf?
				openvdb::Coord head = iter.getCoord() + loop_order.at(ivoxel);

				if (head[0] < 0 || head[1] < 0 || head[2] < 0 || head[0]>7 || head[1]>7 || head[2]>7) {
					continue;
				}

				auto write_offset = leaf.coordToOffset(head);

				//calculate the distance
				//arg(A,B): ret A-B
				// the absolute value trick: abs_mask: 01111111..32bit..1111 x 4
				// _mm_and_ps(abs_mask(), v);
				x_dist_particle_to_sample = _mm_and_ps(absmask, _mm_sub_ps(x_offset_to_center_voxel_center.at(ivoxel), particle_x));
				y_dist_particle_to_sample = _mm_and_ps(absmask, _mm_sub_ps(y_offset_to_center_voxel_center.at(ivoxel), particle_y));
				z_dist_particle_to_sample = _mm_and_ps(absmask, _mm_sub_ps(z_offset_to_center_voxel_center.at(ivoxel), particle_z));

				//the distance to the phi variable
				_mm_store_ss(&tophix, x_dist_particle_to_sample);
				_mm_store_ss(&tophiy, y_dist_particle_to_sample);
				_mm_store_ss(&tophiz, z_dist_particle_to_sample);
				dist_to_phi_sample = dx * std::sqrt(tophix * tophix + tophiy * tophiy + tophiz * tophiz);

				//phi
				//compare 
				float original_sdf = new_sdf_leaf->getValue(write_offset);
				new_sdf_leaf->setValueOn(write_offset, std::min(original_sdf, dist_to_phi_sample - m_particle_radius));

				do {
					//if for [x,y,z]distance_to_sample, the first 3 values are all greater than 1
					//then the weight is 0, so skip it
					uint32_t test = _mm_movemask_ps(_mm_cmpgt_ps(x_dist_particle_to_sample, comp_target));
					if (test == 0b00001111) {
						break;
					}
					test = _mm_movemask_ps(_mm_cmpgt_ps(y_dist_particle_to_sample, comp_target));
					if (test == 0b00001111) {
						break;
					}
					test = _mm_movemask_ps(_mm_cmpgt_ps(z_dist_particle_to_sample, comp_target));
					if (test == 0b00001111) {
						break;
					}

					//the uvw weights trilinear
					//transfer the distance to weight at the 27 voxels
					//(1-dist)
					//the far points now becomes negative
					x_dist_particle_to_sample = _mm_sub_ps(float1x4, x_dist_particle_to_sample);
					y_dist_particle_to_sample = _mm_sub_ps(float1x4, y_dist_particle_to_sample);
					z_dist_particle_to_sample = _mm_sub_ps(float1x4, z_dist_particle_to_sample);

					//turn everything positive or zero
					//now the dist_to_sample is actually the component-wise weight on the voxel
					//time to multiply them together
					x_dist_particle_to_sample = _mm_max_ps(float0x4, x_dist_particle_to_sample);
					y_dist_particle_to_sample = _mm_max_ps(float0x4, y_dist_particle_to_sample);
					z_dist_particle_to_sample = _mm_max_ps(float0x4, z_dist_particle_to_sample);

					//turn them into weights reduce to x
					x_dist_particle_to_sample = _mm_mul_ps(x_dist_particle_to_sample, y_dist_particle_to_sample);
					x_dist_particle_to_sample = _mm_mul_ps(x_dist_particle_to_sample, z_dist_particle_to_sample);
					//}//end for 27 voxel

					////write to the grid
					//for (size_t ivoxel = 0; ivoxel < 27; ivoxel++) {
					alignas(16) float packed_weight[4];

					_mm_storer_ps(packed_weight, x_dist_particle_to_sample);

					openvdb::Vec3f weights{ packed_weight[0],packed_weight[1],packed_weight[2] };
					//write weighted velocity

					openvdb::Vec3f weighted_vel = pvel * weights;
					auto original_weighted_vel = new_vel_leaf->getValue(write_offset);
					new_vel_leaf->setValueOn(write_offset, weighted_vel + original_weighted_vel);
					//new_vel_leaf->modifyValue(write_offset, [&weighted_vel](openvdb::Vec3f& in_out) {in_out += weighted_vel; });
					//write weights
					auto original_weights = new_vel_weights_leaf->getValue(write_offset);
					new_vel_weights_leaf->setValueOn(write_offset, weights + original_weights);
					//new_vel_weights_leaf->modifyValue(write_offset, [&weights](openvdb::Vec3f& in_out) {in_out += weights; });

				} while (false);//test if the velocity weight is zero
			}//end for 27 voxels
		}//end for all particles influencing this leaf
	}//end operator

	//particle radius 
	float m_particle_radius;

	//neighbor leaf pointer cache

	//constant distances to the center voxel
	std::array<__m128, 27> x_offset_to_center_voxel_center;
	std::array<__m128, 27> y_offset_to_center_voxel_center;
	std::array<__m128, 27> z_offset_to_center_voxel_center;
	std::array<openvdb::Coord, 27> loop_order;

	openvdb::FloatGrid::Ptr m_liquid_sdf;
	openvdb::Vec3fGrid::Ptr m_velocity, m_velocity_weight;
	openvdb::points::PointDataGrid::Ptr m_particles;
};
}

namespace {
struct tbbcoordhash {
	std::size_t hash(const openvdb::Coord& c) const {
		return c.hash();
	}

	bool equal(const openvdb::Coord& a, const openvdb::Coord& b) const {
		return a == b;
	}
};
}

namespace {
struct narrow_band_copy_iterator {

	narrow_band_copy_iterator(const std::vector<openvdb::Index>& in_idx_of_source) :
		m_idx_of_source(in_idx_of_source) {
		m_it = 0;
	}

	narrow_band_copy_iterator(const narrow_band_copy_iterator& other) :
		m_idx_of_source(other.m_idx_of_source),
		m_it(other.m_it) {

	}
	operator bool() const { return m_it < m_idx_of_source.size(); }

	narrow_band_copy_iterator& operator++() { ++m_it; return *this; }

	openvdb::Index sourceIndex() const
	{
		return m_idx_of_source[m_it];
	}

	openvdb::Index targetIndex() const
	{
		return m_it;
	}

	//idx in the attribute array
	//[target_idx]=source_idx
	size_t m_it;
	const std::vector<openvdb::Index>& m_idx_of_source;
};
}

void Compressed_FLIP_Object_OpenVDB::narrow_band_particles(openvdb::points::PointDataGrid::Ptr out_nb_particles,
	openvdb::FloatGrid::Ptr out_interior_sdf,
	openvdb::points::PointDataGrid::Ptr in_particles, int nlayer)
{
	ZoneScoped;
	//particles are stored in particle voxels
	//1 dilate particle voxels and mark existing part as inactive to get the surface mask
	//2 dilate the surface mask nlayer-1 to get the erosion mask
	//3 dilate the surface mask nlayer to get the output particle mask
	//4 use the erosion mask to generate interior sdf (every voxel has -0.5dx)
	//5 use the output particle mask to generate the narrow banded particles

	using point_leaf_type = openvdb::points::PointDataTree::LeafNodeType;

	//1.1 dilate particle voxels to get surface mask
	auto surface_mask = openvdb::BoolGrid::create();
	surface_mask->setTree(std::make_shared<openvdb::BoolTree>(in_particles->tree(), /*inactive value*/false,/*active value*/true, openvdb::TopologyCopy()));
	auto original_particle_mask = surface_mask->deepCopy();

	openvdb::tools::dilateActiveValues(surface_mask->tree(), 1,
		openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);

	//1.2 hollow the surface mask
	surface_mask->topologyDifference(*original_particle_mask);

	//in some turbulent areas, there are many small holes in the interior of the liquid
	//these holes should not be marked as surface
	auto fill_holes_lambda = [&](openvdb::BoolTree::LeafNodeType& leaf, openvdb::Index) {
		auto particle_axr = in_particles->getConstUnsafeAccessor();
		auto particle_mask_leaf = original_particle_mask->tree().probeLeaf(leaf.origin());
		for (auto iter = leaf.beginValueOn(); iter; ++iter) {
			//if all its neighbor is inside liquid
			//it is still surrounded by particles
			int void_voxel_count = 0;
			auto itercoord = iter.getCoord();
			for (int ii = -1; ii < 2; ii++) {
				for (int jj = -1; jj < 2; jj++) {
					for (int kk = -1; kk < 2; kk++) {
						if (ii == 0 && jj == 0 && kk == 0) {
							continue;
						}
						if (!particle_axr.isValueOn(itercoord.offsetBy(ii, jj, kk))) {
							void_voxel_count++;
						}
					}
				}
			}
			if (void_voxel_count <= 2) {
				//disable this surface mask, so particles around it wont be in the output
				iter.setValueOff();
				//treat this point as interior point in the particle mask
				if (particle_mask_leaf) {
					particle_mask_leaf->setValueOn(iter.offset());
				}
			}
		}
	};
	openvdb::tree::LeafManager<openvdb::BoolTree> surface_mask_man(surface_mask->tree());
	surface_mask_man.foreach(fill_holes_lambda);

	//2 dilate the surface mask nlayer-1 to get the erosion mask
	auto erosion_mask = surface_mask->deepCopy();
	openvdb::tools::dilateActiveValues(erosion_mask->tree(), nlayer - 1,
		openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);

	//3 dilate the erosion mask to get the output particle mask
	auto output_particle_mask = erosion_mask->deepCopy();
	openvdb::tools::dilateActiveValues(output_particle_mask->tree(), 1,
		openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);

	//4 use the erosion mask to generate interior sdf (every voxel has -0.5dx)
	float dx = in_particles->transform().voxelSize()[0];

	auto interior_sdf = openvdb::FloatGrid::create(0.5 * dx);
	auto interior_mask = original_particle_mask->deepCopy();
	interior_mask->topologyDifference(*erosion_mask);
	interior_sdf->setTree(std::make_shared<openvdb::FloatTree>(interior_mask->tree(),
		/*inactive value*/0.5f * dx,/*active value*/-0.5f * dx, openvdb::TopologyCopy()));

	//5 use the output particle mask to generate the narrow banded particles
	auto nb_particles = openvdb::points::PointDataGrid::create();
	nb_particles->setTree(std::make_shared<openvdb::points::PointDataTree>(output_particle_mask->tree(),
		/*background value*/0, openvdb::TopologyCopy()));

	std::vector<int> leaf_empty;
	leaf_empty.assign(nb_particles->tree().leafCount(), 0);
	std::vector<openvdb::points::PointDataTree::LeafNodeType*> point_leaves;
	point_leaves.reserve(nb_particles->tree().leafCount());
	nb_particles->tree().getNodes(point_leaves);

	auto populate_nb_particle_lambda = [&in_particles, &leaf_empty]
	(openvdb::points::PointDataTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		auto original_leaf = in_particles->tree().probeConstLeaf(leaf.origin());
		if (!original_leaf) {
			leaf_empty[leafpos] = 1;
			return;
		}

		if (original_leaf->getLastValue() == 0) {
			//original leaf is empty, possibly not having any attribute sets
			leaf_empty[leafpos] = 1;
			return;
		}

		//write idx ends
		int emitted = 0;
		std::vector<openvdb::Index> original_particle_idx;

		for (auto offset = 0; offset < leaf.size(); ++offset) {
			auto idxend = original_leaf->getValue(offset);
			auto idxbegin = 0;
			if (offset != 0) {
				idxbegin = original_leaf->getValue(offset - 1);
			}

			//emit if mask on
			if (leaf.isValueOn(offset)) {
				for (auto it = idxbegin; it < idxend; ++it) {
					original_particle_idx.push_back(it);
				}
				emitted += idxend - idxbegin;
			}
			leaf.setOffsetOnly(offset, emitted);
		}

		//copy the attributes
		leaf.replaceAttributeSet(new openvdb::points::AttributeSet(original_leaf->attributeSet(), emitted), true);

		for (const auto& it : original_leaf->attributeSet().descriptor().map()) {
			const auto attributeIndex = static_cast<openvdb::Index>(it.second);

			auto& target_array = leaf.attributeArray(attributeIndex);

			narrow_band_copy_iterator copyiter(original_particle_idx);

			target_array.copyValues(original_leaf->attributeArray(attributeIndex), copyiter);
		}
	};//end populate narrow band particles lambda
	auto nb_manager = openvdb::tree::LeafManager<openvdb::points::PointDataTree>(nb_particles->tree());
	nb_manager.foreach(populate_nb_particle_lambda, true);

	//write to the output
	interior_sdf->pruneGrid();
	//nb_particles->pruneGrid();

	for (size_t i = 0; i < leaf_empty.size(); ++i) {
		if (leaf_empty[i]) {
			delete nb_particles->tree().stealNode<openvdb::points::PointDataTree::LeafNodeType>(point_leaves[i]->origin(), 0, false);
		}
	}


	out_interior_sdf->setTree(interior_sdf->treePtr());
	out_interior_sdf->setTransform(in_particles->transform().copy());
	out_interior_sdf->setName("surface");
	out_interior_sdf->setGridClass(openvdb::GRID_LEVEL_SET);
	out_nb_particles->setTree(nb_particles->treePtr());
	out_nb_particles->setTransform(in_particles->transform().copy());
	out_nb_particles->setName("Particles");
}

FLIP_Solver_OpenVDB::FLIP_Solver_OpenVDB(const FLIP_Object_OpenVDB& in_object, const FLIP_Solver_OpenVDB::SolverParameters& in_parm)
{
	using  namespace openvdb::tools::local_util;
	m_dx = static_cast<float>(in_object.m_particles->voxelSize()[0]);

	m_solver_parameters = in_parm;

	//construct the index minimum and maximum that defines the simulation domain
	m_domain_index_begin = floorVec3(in_parm.domain_min / m_dx);
	m_domain_index_end = ceilVec3(in_parm.domain_max / m_dx);

	init_transforms();
	init_grids();
	m_packed_velocity = in_object.m_velocity;
	m_packed_vel_after_p2g = in_object.m_velocity.deepCopy();
}

FLIP_Solver_OpenVDB::~FLIP_Solver_OpenVDB()
{
}

void FLIP_Solver_OpenVDB::solve(FLIP_Object_OpenVDB& in_out_object, float in_dt)
{
	const float target_time = in_dt;
	float stepped = 0.f;
	int sub_step_count = 0;

	while (stepped < target_time) {
		auto cfl_dt = m_solver_parameters.cfl * cfl();
		if (m_solver_parameters.simulation_method == SimulationMethod::ADVECTION_REFLECITON) {
			cfl_dt *= 2.0f;
		}
		float estimated_steps = std::ceil((target_time - stepped) / cfl_dt);
		float estimated_dt = (target_time - stepped) / estimated_steps * 1.1f;
		float dt = std::min(estimated_dt, cfl_dt);

		float reaching_dt = stepped + dt;

		if (stepped + cfl_dt > target_time) {
			reaching_dt = target_time;
		}

		if (stepped + dt > target_time) {
			reaching_dt = target_time;
		}

		dt = reaching_dt - stepped;
		stepped = reaching_dt;
		printf("substep:%d, estimated_left:%d, toward %d/100 \n", sub_step_count, int(estimated_steps), (int)std::floor(stepped / target_time * 100.0f));
		substep(in_out_object, dt);
		sub_step_count++;
	}
}

void FLIP_Solver_OpenVDB::custom_move_points_and_set_flip_vel(
	openvdb::points::PointDataGrid::Ptr in_out_points,
	packed_FloatGrid3 in_velocity_field,
	packed_FloatGrid3 in_velocity_field_to_be_advected,
	packed_FloatGrid3 in_old_velocity,
	openvdb::FloatGrid::Ptr in_solid_sdf,
	openvdb::Vec3fGrid::Ptr in_solid_vel,
	float PIC_component, float dt, int RK_order)
{
	ZoneScoped;
	if (!in_out_points) {
		return;
	}

	float dx = in_out_points->transform().voxelSize()[0];
	bool has_valid_solid = true;
	if (!in_solid_sdf) {
		auto corner_transform = openvdb::math::Transform::createLinearTransform(dx);
		in_solid_vel = openvdb::Vec3fGrid::create(openvdb::Vec3f(0)); in_solid_vel->setTransform(corner_transform);

		corner_transform->postTranslate(openvdb::Vec3d{ -0.5,-0.5,-0.5 }*double(dx));
		in_solid_sdf = openvdb::FloatGrid::create(3 * dx); in_solid_sdf->setTransform(corner_transform);
		has_valid_solid = false;
	}

	//prepare voxel center solid outward normal direction
	//in case a particle falls into solid during advection, this will help pushing the particle out.
	openvdb::Vec3fGrid::Ptr voxel_center_solid_normal = openvdb::Vec3fGrid::create(openvdb::Vec3f(0));
	openvdb::FloatGrid::Ptr voxel_center_solid_vn = openvdb::FloatGrid::create(0);

	if (has_valid_solid) {
		voxel_center_solid_normal->setTree(std::make_shared<openvdb::Vec3fTree>(
			in_solid_sdf->tree(), /*bgval*/openvdb::Vec3f(0), openvdb::TopologyCopy()));
	}

	//openvdb::tools::dilateActiveValues(voxel_center_solid_normal->tree(), 5, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX);
	voxel_center_solid_normal->setTransform(in_out_points->transformPtr());
	voxel_center_solid_normal->setName("solidnormal");

	voxel_center_solid_vn->setTree(std::make_shared<openvdb::FloatTree>(
		voxel_center_solid_normal->tree(), /*bgval*/0.f, openvdb::TopologyCopy()));
	voxel_center_solid_vn->setTransform(in_out_points->transformPtr());
	voxel_center_solid_vn->setName("vn");

	Eigen::MatrixXf AT;
	Eigen::Vector4f invATA;
	AT.resize(4, 8);
	for (int i = 0; i < 8; i++) {
		int a = i / 4;
		int b = (i - a * 4) / 2;
		int c = i - a * 4 - b * 2;
		AT.col(i) = Eigen::Vector4f{ a - 0.5f,b - 0.5f,c - 0.5f, 1.0f };
	}

	AT *= dx;
	invATA.setZero(4);
	invATA(0) = 0.5f; invATA(1) = 0.5f; invATA(2) = 0.5f; invATA(3) = 0.125f;
	invATA *= 1.0f / (dx * dx);

	auto set_voxel_center_solid_grad_and_vn = [&](openvdb::Vec3fTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		auto vnleaf = voxel_center_solid_vn->tree().probeLeaf(leaf.origin());
		auto sdfaxr = in_solid_sdf->getConstUnsafeAccessor();
		auto velaxr = in_solid_vel->getConstUnsafeAccessor();

		Eigen::VectorXf data;
		data.setZero(8);

		for (auto iter = leaf.beginValueOn(); iter; ++iter) {
			auto gcoord = iter.getCoord();
			//calculate the sdf at this voxel center
			auto& i = gcoord[0]; auto& j = gcoord[1]; auto& k = gcoord[2];
			data[0] = sdfaxr.getValue(gcoord);//000
			k++;  data[1] = sdfaxr.getValue(gcoord);//001
			j++;  data[3] = sdfaxr.getValue(gcoord);//011
			k--;  data[2] = sdfaxr.getValue(gcoord);//010
			i++;  data[6] = sdfaxr.getValue(gcoord);//110
			k++;  data[7] = sdfaxr.getValue(gcoord);//111
			j--;  data[5] = sdfaxr.getValue(gcoord);//101
			k--;  data[4] = sdfaxr.getValue(gcoord);//100

			Eigen::VectorXf abcd;
			abcd = invATA.array() * (AT * data).array();

			//d is the sdf at center
			if (abcd[3] < 1.0f) {
				openvdb::Vec3f abc{ abcd[0], abcd[1], abcd[2] };
				abc.normalize();
				leaf.setValueOn(iter.offset(), abc);

				//solid velocity on this direction
				auto solidvel = velaxr.getValue(gcoord);
				vnleaf->setValueOn(iter.offset(), solidvel.dot(abc));
			}
			else {
				leaf.setValueOff(iter.offset());
				vnleaf->setValueOff(iter.offset());
			}
		}//for all on voxels
	};//end set_voxel_center_solid_grad_and_vn
	auto voxel_center_solid_normalman = openvdb::tree::LeafManager<openvdb::Vec3fTree>(voxel_center_solid_normal->tree());
	if (has_valid_solid) {
		voxel_center_solid_normalman.foreach(set_voxel_center_solid_grad_and_vn);
	}

	voxel_center_solid_normal->pruneGrid();
	voxel_center_solid_vn->pruneGrid();

	//openvdb::io::File("vn.vdb").write({ voxel_center_solid_vn, voxel_center_solid_normal });

	std::vector<openvdb::points::PointDataTree::LeafNodeType*> particle_leaves;
	particle_leaves.reserve(in_out_points->tree().leafCount());
	in_out_points->tree().getNodes(particle_leaves);

	auto reducer = std::make_unique<point_to_counter_reducer>(
		dt, dx,
		in_velocity_field, in_velocity_field_to_be_advected, in_old_velocity,
		*in_solid_sdf, *voxel_center_solid_normal, *voxel_center_solid_vn,
		PIC_component, particle_leaves, *in_out_points, RK_order);
	tbb::parallel_reduce(tbb::blocked_range<openvdb::Index>(0, particle_leaves.size(), 1), *reducer);

	//compose the result
	//create new particle leaf origins linear vector
	openvdb::points::PointDataGrid::Ptr new_particle_grid = openvdb::points::PointDataGrid::create();
	std::vector<openvdb::Coord> new_particle_leaf_origins;
	std::vector<point_to_counter_reducer::to_oi_olp_container_type*> tuple_vec_raw_ptrs;
	new_particle_leaf_origins.reserve(reducer->toffset_oindex_oleafpos_hashmap.size());
	tuple_vec_raw_ptrs.reserve(reducer->toffset_oindex_oleafpos_hashmap.size());
	for (auto iter = reducer->toffset_oindex_oleafpos_hashmap.begin(); iter != reducer->toffset_oindex_oleafpos_hashmap.end(); ++iter) {
		new_particle_leaf_origins.push_back(iter->first);
		tuple_vec_raw_ptrs.push_back(iter->second.get());
	}

	//concurrently create the new tree
	std::vector<openvdb::points::PointDataTree::LeafNodeType*> new_particle_leaves;
	new_particle_leaves.assign(new_particle_leaf_origins.size(), nullptr);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, new_particle_leaf_origins.size()), [&](const tbb::blocked_range<size_t>& r) {
		for (size_t i = r.begin(); i != r.end(); ++i) {
			new_particle_leaves[i] = new openvdb::points::PointDataTree::LeafNodeType(new_particle_leaf_origins[i]);
		}});
	for (auto leaf : new_particle_leaves) {
		new_particle_grid->tree().addLeaf(leaf);
	}

	auto newTree_leafman{ openvdb::tree::LeafManager<openvdb::points::PointDataTree>(new_particle_grid->tree()) };

	auto pnamepair = PositionAttribute::attributeType();
	auto position_attribute_descriptor = openvdb::points::AttributeSet::Descriptor::create(pnamepair);

	auto vnamepair = VelocityAttribute::attributeType();
	auto pv_attribute_descriptor = position_attribute_descriptor->duplicateAppend("v", vnamepair);


	//set up the new tree, converting each new leaf into a valid point data grid leaf
	//note the leafpos here is not the same from the leaf manager, because it is extracted from an unordered map
	auto set_new_attribute_list = [&](const tbb::blocked_range<size_t>& r) {
		using  namespace openvdb::tools::local_util;

		for (size_t leafpos = r.begin(); leafpos != r.end(); ++leafpos) {
			auto& leaf = *new_particle_leaves[leafpos];
			const auto& toffset_oidx_oleafpos_vec = *tuple_vec_raw_ptrs[leafpos];
			std::vector<openvdb::PointDataIndex32> index_ends; index_ends.resize(leaf.size(), 0);

			//first the index_ends contains the number of particles in this voxel
			//only fill each voxel up to 16 particles
			for (auto& tuple : toffset_oidx_oleafpos_vec) {
				if (index_ends[tuple.target_offset] < 32) {
					index_ends[tuple.target_offset] = index_ends[tuple.target_offset] + 1;
				}
			}
			for (auto offset = 1; offset < leaf.size(); ++offset) {
				index_ends[offset] = index_ends[offset - 1] + index_ends[offset];
			}

			//create the attribute set
			auto local_pv_descriptor = pv_attribute_descriptor;
			leaf.initializeAttributes(position_attribute_descriptor, index_ends.back());
			leaf.appendAttribute(leaf.attributeSet().descriptor(), local_pv_descriptor, 1);

			leaf.setOffsets(index_ends);

			//turn the index_ends back to voxel count at each voxel again
			auto& npar_to_be_filled = index_ends;
			for (auto offset = leaf.size() - 1; offset >= 1; --offset) {
				npar_to_be_filled[offset] = npar_to_be_filled[offset] - npar_to_be_filled[offset - 1];
			}

			//set the positions and velocities
			//get the new attribute arrays
			openvdb::points::AttributeArray& posarray = leaf.attributeArray("P");
			openvdb::points::AttributeArray& varray = leaf.attributeArray("v");

			// Create read handles for position and velocity
			openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_Object_OpenVDB::PositionCodec> posWHandle(posarray);
			openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_Object_OpenVDB::VelocityCodec> vWHandle(varray);

			openvdb::Vec3i pcoord;
			int writing_offset;
			int writer_index;

			//move from the original leaves
			int Npar = toffset_oidx_oleafpos_vec.size();
			for (size_t i = 0; i < Npar; i++) {
				writing_offset = toffset_oidx_oleafpos_vec[i].target_offset;
				if (npar_to_be_filled[writing_offset] > 0) {
					//potentially more than allowed particles want to be filled into the target voxel
					//however they wont be allowed if it is already filled with sufficient amount
					writer_index = leaf.getValue(writing_offset) - npar_to_be_filled[writing_offset];
					npar_to_be_filled[writing_offset] = npar_to_be_filled[writing_offset] - 1;
					posarray.set(writer_index, particle_leaves[toffset_oidx_oleafpos_vec[i].original_leaf_pos]->attributeArray("P"), toffset_oidx_oleafpos_vec[i].original_index);
					varray.set(writer_index, particle_leaves[toffset_oidx_oleafpos_vec[i].original_leaf_pos]->attributeArray("v"), toffset_oidx_oleafpos_vec[i].original_index);
				}
			}//end for all recorded particles in this leaf
		}//end for each new leaf
	};//end set new particle leaves

	//printf("compose_start_for_each\n");
	tbb::parallel_for(tbb::blocked_range<size_t>(0, new_particle_leaves.size()), set_new_attribute_list);

	new_particle_grid->setName("new_counter_grid");
	new_particle_grid->setTransform(in_out_points->transformPtr());

	in_out_points->setTree(new_particle_grid->treePtr());
}

void FLIP_Solver_OpenVDB::emit_liquid(openvdb::points::PointDataGrid::Ptr in_out_particles, const sdf_vel_pair& in_sdf_vel, const openvdb::Vec3f& in_emit_world_min, const openvdb::Vec3f& in_emit_world_max)
{
	using  namespace openvdb::tools::local_util;
	if (!in_sdf_vel.m_activated) {
		return;
	}
	//alias
	auto in_sdf = in_sdf_vel.m_sdf;
	auto in_vel = in_sdf_vel.m_vel;

	//the input is assumed to be narrowband boxes
	auto inputbbox = openvdb::CoordBBox();
	auto is_valid = in_sdf->tree().evalLeafBoundingBox(inputbbox);
	if (!is_valid) {
		return;
	}

	//bounding box in the world coordinates
	auto worldinputbbox = openvdb::BBoxd(in_sdf->indexToWorld(inputbbox.min()), in_sdf->indexToWorld(inputbbox.max()));
	for (int ic = 0; ic < 3; ic++) {
		if (worldinputbbox.min()[ic] < in_emit_world_min[ic]) {
			worldinputbbox.min()[ic] = in_emit_world_min[ic];
		}
		if (worldinputbbox.max()[ic] > in_emit_world_max[ic]) {
			worldinputbbox.max()[ic] = in_emit_world_max[ic];
		}
	}

	//map the bounding box to the particle index space
	auto loopbegin = floorVec3(in_out_particles->worldToIndex(worldinputbbox.min())) - openvdb::Vec3i(8);
	auto loopend = ceilVec3(in_out_particles->worldToIndex(worldinputbbox.max())) + openvdb::Vec3i(8);

	auto tempgrid = openvdb::BoolGrid::create();
	tempgrid->setTransform(in_out_particles->transformPtr());
	auto templeaf = tempgrid->tree().touchLeaf(openvdb::Coord(loopbegin));

	loopbegin = templeaf->origin().asVec3i() / 8;
	templeaf = tempgrid->tree().touchLeaf(openvdb::Coord(loopend));
	loopend = templeaf->origin().asVec3i() / 8;

	using point_leaf_type = openvdb::points::PointDataTree::LeafNodeType;
	tbb::concurrent_hash_map<openvdb::Coord, point_leaf_type*, tbbcoordhash> target_point_leaves;

	auto pispace_range = tbb::blocked_range3d<int, int, int>(loopbegin.x(), loopend.x(), loopbegin.y(), loopend.y(), loopbegin.z(), loopend.z());

	float dx = in_out_particles->transform().voxelSize()[0];
	auto mark_active_point_leaves = [&](const tbb::blocked_range3d<int, int, int>& r) {
		auto solid_axr{ in_sdf->getConstUnsafeAccessor() };

		for (int i = r.pages().begin(); i < r.pages().end(); i += 1) {
			for (int j = r.rows().begin(); j < r.rows().end(); j += 1) {
				for (int k = r.cols().begin(); k < r.cols().end(); k += 1) {
					//test if any of the corner of this box touches the 
					int has_solid = 0;
					int i8 = i * 8, j8 = j * 8, k8 = k * 8;

					for (int ii = 0; ii <= 8 && !has_solid; ii += 1) {
						for (int jj = 0; jj <= 8 && !has_solid; jj += 1) {
							for (int kk = 0; kk <= 8 && !has_solid; kk += 1) {
								auto particle_idx_pos = openvdb::Vec3i(i8 + ii, j8 + jj, k8 + kk) - openvdb::Vec3f(0.5f);
								has_solid = (openvdb::tools::BoxSampler::sample(solid_axr, in_sdf->worldToIndex(in_out_particles->indexToWorld(particle_idx_pos))) < 0);
							}
						}
					}

					if (has_solid) {
						auto floor_leaf = in_out_particles->tree().probeLeaf(openvdb::Coord(i8, j8, k8));
						if (floor_leaf) {
							target_point_leaves.insert(std::make_pair(floor_leaf->origin(), floor_leaf));
						}
						else {
							auto newleaff = new openvdb::points::PointDataTree::LeafNodeType(openvdb::Coord(i8, j8, k8), 0, true);
							auto new_origin = newleaff->origin();
							newleaff->setValuesOff();
							bool success = target_point_leaves.insert(std::make_pair(newleaff->origin(), newleaff));
							if (!success) {
								delete newleaff;
							}
						}//end else found existing particle leaf
					}//end if touch solid
				}//loop k
			}//loop j
		}//loop i
	};//end mark active point elaves
	tbb::parallel_for(pispace_range, mark_active_point_leaves);

	//for parallel process
	std::vector<point_leaf_type*> leafarray;
	for (auto& i : target_point_leaves) {
		leafarray.push_back(i.second);
	}
	//printf("touched_leafs:%zd\n", leafarray.size());

	//std::atomic<size_t> added_particles{ 0 };

	auto pnamepair = PositionAttribute::attributeType();
	auto position_attribute_descriptor = openvdb::points::AttributeSet::Descriptor::create(pnamepair);

	auto vnamepair = VelocityAttribute::attributeType();
	auto pv_attribute_descriptor = position_attribute_descriptor->duplicateAppend("v", vnamepair);

	//use existing descriptors if we already have particles
	if (!in_out_particles->empty()) {
		for (auto it = in_out_particles->tree().cbeginLeaf(); it; ++it) {
			if (it->getLastValue() != 0) {
				//a non-empty leaf with attributes
				pv_attribute_descriptor = it->attributeSet().descriptorPtr();
				break;
			}
		}
	}

	//seed each leaf
	auto seed_leaf = [&](const tbb::blocked_range<size_t>& r) {
		auto in_sdf_axr{ in_sdf->getConstUnsafeAccessor() };
		auto in_vel_axr{ in_vel->getConstUnsafeAccessor() };
		//float sdf_threshold = -dx * 0.55;
		float sdf_threshold = -dx * 0.f;

		std::random_device device;
		std::mt19937 generator(/*seed=*/device());
		std::uniform_real_distribution<> distribution(-0.5, 0.5);

		for (auto ileaf = r.begin(); ileaf < r.end(); ++ileaf) {
			auto& leaf = *leafarray[ileaf];
			custom_vector_type<openvdb::Vec3f> new_pos, new_vel;
			std::vector<openvdb::PointDataIndex32> new_idxend;

			if (leaf.isValueMaskOn()) {
				continue;
			}

			//check if the last element is zero
			//if so, it's a new leaf
			if (leaf.getLastValue() == 0) {
				//initialize the attribute descriptor
				auto local_pv_descriptor = pv_attribute_descriptor;
				leaf.initializeAttributes(position_attribute_descriptor, 0);
				leaf.appendAttribute(leaf.attributeSet().descriptor(), local_pv_descriptor, 1);
			}

			//reader
			const openvdb::points::AttributeArray& positionArray =
				leaf.attributeArray("P");
			// Extract the velocity attribute from the leaf by name (v is velocity).
			const openvdb::points::AttributeArray& velocityArray =
				leaf.attributeArray("v");
			auto p_handle_ptr = openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_Object_OpenVDB::PositionCodec>::create(positionArray);
			auto v_handle_ptr = openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_Object_OpenVDB::VelocityCodec>::create(velocityArray);

			openvdb::Index32 emitted_particle = 0;
			for (auto offset = 0; offset < leaf.SIZE; ++offset) {
				openvdb::Index32 idxbegin = 0;
				if (offset != 0) {
					idxbegin = leaf.getValue(offset - 1);
				}
				openvdb::Index32 idxend = leaf.getValue(offset);


				//used to indicate sub_voxel(8 cells) occupancy by particles
				unsigned char sub_voxel_occupancy = 0;

				//first emit original particles
				openvdb::Index32 this_voxel_emitted = 0;
				for (auto idx = idxbegin; idx < idxend; ++idx) {
					auto p = p_handle_ptr->get(idx);
					unsigned char sv_pos = ((p[2] > 0.f) << 2) | ((p[1] > 0.f) << 1) | (p[0] > 0.f);
					//only emit uniformly, otherwise skip it
					if (sub_voxel_occupancy & (1 << sv_pos)) {
						//that bit is active, there is already an particle
						//skip it
						//continue;
					}
					sub_voxel_occupancy |= (1 << sv_pos);
					new_pos.push_back(p);
					new_vel.push_back(v_handle_ptr->get(idx));
					emitted_particle++;
					this_voxel_emitted++;
				}

				//emit to fill the sdf
				auto voxelwpos = in_out_particles->indexToWorld(leaf.offsetToGlobalCoord(offset));
				auto voxelipos = in_sdf->worldToIndex(voxelwpos);
				if (openvdb::tools::BoxSampler::sample(in_sdf_axr, voxelipos) < dx) {
					const int max_emit_trial = 32;
					for (int trial = 0; this_voxel_emitted < 8 && trial < max_emit_trial; trial++) {
						openvdb::Vec3d particle_pipos{ distribution(generator) ,distribution(generator) ,distribution(generator) };
						auto& p = particle_pipos;
						unsigned char sv_pos = ((p[2] > 0) << 2) | ((p[1] > 0) << 1) | (p[0] > 0);
						//only emit uniformly, otherwise skip it
						if (sub_voxel_occupancy & (1 << sv_pos)) {
							//that bit is active, there is already an particle
							//skip it
							continue;
						}
						auto particlewpos = in_out_particles->indexToWorld(particle_pipos) + voxelwpos;
						auto particle_sdfipos = in_sdf->worldToIndex(particlewpos);
						if (openvdb::tools::BoxSampler::sample(in_sdf_axr, particle_sdfipos) < sdf_threshold) {
							sub_voxel_occupancy |= 1 << sv_pos;
							new_pos.push_back(particle_pipos);
							new_vel.push_back(openvdb::tools::BoxSampler::sample(in_vel_axr, particle_sdfipos));
							emitted_particle++;
							this_voxel_emitted++;
							//added_particles++;
						}
					}//end for 16 emit trials
				}//if voxel inside emitter

				new_idxend.push_back(emitted_particle);
			}//end emit original particles and new particles

			//set the new index ends, attributes
			//initialize the attribute descriptor
			auto local_pv_descriptor = pv_attribute_descriptor;
			leaf.initializeAttributes(position_attribute_descriptor, emitted_particle);
			leaf.appendAttribute(leaf.attributeSet().descriptor(), local_pv_descriptor, 1);
			leaf.setOffsets(new_idxend);

			//set the positions and velocities
			//get the new attribute arrays
			openvdb::points::AttributeArray& posarray = leaf.attributeArray("P");
			openvdb::points::AttributeArray& varray = leaf.attributeArray("v");

			// Create read handles for position and velocity
			openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_Object_OpenVDB::PositionCodec> posWHandle(posarray);
			openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_Object_OpenVDB::VelocityCodec> vWHandle(varray);

			for (auto iter = leaf.beginIndexOn(); iter; ++iter) {
				posWHandle.set(*iter, new_pos[*iter]);
				vWHandle.set(*iter, new_vel[*iter]);
			}//end for all on voxels
		}//end for range leaf
	};//end seed particle 

	tbb::parallel_for(tbb::blocked_range<size_t>(0, leafarray.size()), seed_leaf);
	//printf("added particles:%zd\n", size_t(added_particles));
	//printf("original leaf count:%d\n", m_particles->tree().leafCount());
	for (auto leaf : leafarray) {
		if (leaf != in_out_particles->tree().probeLeaf(leaf->origin())) {
			in_out_particles->tree().addLeaf(leaf);
		}
	}
	//openvdb::io::File("dbg.vdb").write({ m_particles });
	//printf("new leafcount:%d\n", m_particles->tree().leafCount());
	in_out_particles->pruneGrid();
}


FLIP_Solver_OpenVDB::SolverParameters::SolverParameters()
{
	domain_min = openvdb::Vec3f(-10.0f);
	domain_max = openvdb::Vec3f(10.0f);
	cfl = 3.0f;
	PIC_component = 0.05f;
	simulation_method = FLIP_Solver_OpenVDB::SimulationMethod::ADVECTION_PROJECTION;
	domain_boundary_type = FLIP_Solver_OpenVDB::DomainBoundaryType::SINK;
	enable_viscosity = FLIP_Solver_OpenVDB::ViscosityOption::OFF;
	particle_integration_method = FLIP_Solver_OpenVDB::ParticleIntegrationMethod::RK2;
	viscosity_solver_accuracy = 1e-4f;
	pressure_solver_accuracy = 1e-7f;
	boundary_friction_coefficient = 0.f;
	gravity = openvdb::Vec3f(0.f, -9.81f, 0.f);
}

void FLIP_Solver_OpenVDB::particle_to_grid(packed_FloatGrid3 out_velocity, openvdb::FloatGrid::Ptr out_liquid_sdf, openvdb::points::PointDataGrid::Ptr in_particles, const float in_particle_radius)
{
	ZoneScoped;
	//allocate the tree for transfered velocity and liquid phi
	auto unweignted_velocity = openvdb::Vec3fGrid::create();
	unweignted_velocity->setTransform(in_particles->transformPtr());
	unweignted_velocity->setName("Velocity");
	unweignted_velocity->setGridClass(openvdb::GridClass::GRID_STAGGERED);
	unweignted_velocity->setTree(std::make_shared<openvdb::Vec3fTree>(
		in_particles->tree(), openvdb::Vec3f{ 0 }, openvdb::TopologyCopy()));
	openvdb::tools::dilateActiveValues(unweignted_velocity->tree(), 1, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);
	//velocity weights

	auto velocity_weights = unweignted_velocity->deepCopy();

	out_liquid_sdf->setTransform(in_particles->transformPtr());
	out_liquid_sdf->setTree(std::make_shared<openvdb::FloatTree>(
		unweignted_velocity->tree(), out_liquid_sdf->background(), openvdb::TopologyCopy()));


	p2g_collector collector_op{ out_liquid_sdf,
			unweignted_velocity,
			velocity_weights,
			in_particles,
			in_particle_radius };

	auto vleafman = openvdb::tree::LeafManager<openvdb::Vec3fTree>(unweignted_velocity->tree());

	vleafman.foreach(collector_op, true);

	openvdb::tree::LeafManager<openvdb::Vec3fGrid::TreeType> velocity_grid_manager(unweignted_velocity->tree());

	out_velocity.from_vec3(unweignted_velocity, true);

	auto velocity_normalizer = normalize_p2g_velocity(velocity_weights, out_velocity);

	velocity_grid_manager.foreach(velocity_normalizer, true, 1);

	//add another layer of distance information for our sdf
	auto airmask = openvdb::BoolGrid::create();
	airmask->setTree(std::make_shared<openvdb::BoolTree>(out_liquid_sdf->tree(), false, openvdb::TopologyCopy()));
	openvdb::tools::dilateActiveValues(airmask->tree(), 1, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);

	airmask->tree().topologyDifference(out_liquid_sdf->tree());
	airmask->pruneGrid();

	float dx = static_cast<float>(in_particles->voxelSize()[0]);
	//use the particle information of the 2 ring neighbor of the airmask voxel to determine the missing liquid sdf
	//in the air
	auto air_sdf_deducer = [&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		auto air_leaf = airmask->tree().probeConstLeaf(leaf.origin());
		if (!air_leaf) {
			return;
		}
		auto liquid_axr{ out_liquid_sdf->getConstUnsafeAccessor() };

		for (auto air_iter = air_leaf->beginValueOn(); air_iter; ++air_iter) {

			//if it has neighbor negative liquid sdf, means it needs help
			bool has_neighbor_liquid = false;

			float new_sdf = leaf.getValue(air_iter.offset());

			//for its 3*3*3 neighbor, detect potential liquid voxel
			for (int i_neib = 0; i_neib < 6; i_neib++) {
				int component = i_neib / 2;
				int positive_dir = (i_neib % 2 == 0);
				auto at_coord = air_iter.getCoord();
				if (positive_dir) {
					at_coord[component]++;
				}
				else {
					at_coord[component]--;
				}

				float neib_liquid_sdf = liquid_axr.getValue(at_coord);
				if (neib_liquid_sdf < 0) {
					has_neighbor_liquid = true;
					new_sdf = std::min(new_sdf, dx + neib_liquid_sdf);
				}
			}

			if (!has_neighbor_liquid) {
				leaf.setValueOff(air_iter.offset());
				continue;
			}
			else {
				leaf.setValueOn(air_iter.offset(), new_sdf);
			}
			continue;

			leaf.setValueOn(air_iter.offset(), new_sdf);
		}//end for each air voxel
	};//end air_sdf deducer

	openvdb::tools::dilateActiveValues(out_liquid_sdf->tree(), 1, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);
	auto sdfman = openvdb::tree::LeafManager<openvdb::FloatTree>(out_liquid_sdf->tree());
	sdfman.foreach(air_sdf_deducer);
}

openvdb::Vec3fGrid::Ptr FLIP_Solver_OpenVDB::calculate_face_weights(openvdb::FloatGrid::Ptr in_liquid_sdf, openvdb::FloatGrid::Ptr in_solid_sdf)
{
	auto result_face_weight = openvdb::Vec3fGrid::create(openvdb::Vec3f{ 1.0f });
	result_face_weight->setTree(std::make_shared<openvdb::Vec3fTree>(
		in_liquid_sdf->tree(), openvdb::Vec3f{ 1.0f }, openvdb::TopologyCopy()));
	openvdb::tools::dilateActiveValues(result_face_weight->tree(), 2, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);
	result_face_weight->setName("Face_Weights");
	result_face_weight->setTransform(in_liquid_sdf->transformPtr());
	result_face_weight->setGridClass(openvdb::GridClass::GRID_STAGGERED);
	auto set_face_weight_op = [&](openvdb::Vec3fTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		auto solid_axr{ in_solid_sdf->getConstUnsafeAccessor() };
		//solid sdf
		float ssdf[2][2][2];
		openvdb::Vec3f uvwweight{ 1.f };
		for (auto offset = 0; offset < leaf.SIZE; offset++) {
			if (leaf.isValueMaskOff(offset)) {
				continue;
			}
			for (int ii = 0; ii < 2; ii++) {
				for (int jj = 0; jj < 2; jj++) {
					for (int kk = 0; kk < 2; kk++) {
						ssdf[ii][jj][kk] = solid_axr.getValue(leaf.offsetToGlobalCoord(offset) + openvdb::Coord{ ii,jj,kk });
					}
				}
			}//end retrieve eight solid sdfs on the voxel corners

			//fraction_inside(bl,br,tl,tr)
			//tl-----tr
			//|      |
			//|      |
			//bl-----br

			//look from positive x direction

			//   ^z
			//(0,0,1)------(0,1,1)
			//   |            |
			//(0,0,0)------(0,1,0)>y
			//uweight
			uvwweight[0] = 1.0f - fraction_inside(
				ssdf[0][0][0],
				ssdf[0][1][0],
				ssdf[0][0][1],
				ssdf[0][1][1]);
			uvwweight[0] = std::max(0.f, std::min(uvwweight[0], 1.f));

			//look from positive y direction
			//   ^x              
			//(1,0,0)------(1,0,1)
			//   |            |
			//(0,0,0)------(0,0,1)>z
			//vweight
			uvwweight[1] = 1.0f - fraction_inside(
				ssdf[0][0][0],
				ssdf[0][0][1],
				ssdf[1][0][0],
				ssdf[1][0][1]);
			uvwweight[1] = std::max(0.f, std::min(uvwweight[1], 1.f));

			//look from positive z direction
			//   ^y              
			//(0,1,0)------(1,1,0)
			//   |            |
			//(0,0,0)------(1,0,0)>x
			//wweight
			uvwweight[2] = 1.0f - fraction_inside(
				ssdf[0][0][0],
				ssdf[1][0][0],
				ssdf[0][1][0],
				ssdf[1][1][0]);
			uvwweight[2] = std::max(0.f, std::min(uvwweight[2], 1.f));
			leaf.setValueOn(offset, uvwweight);
		}//end for all offset
	};//end set face weight op

	auto leafman = openvdb::tree::LeafManager<openvdb::Vec3fTree>(result_face_weight->tree());
	leafman.foreach(set_face_weight_op);
	return result_face_weight;
}

void FLIP_Solver_OpenVDB::init_grids()
{
	//initialize all grids
	m_pressure = openvdb::FloatGrid::create(float(0));
	m_pressure->setTransform(m_voxel_center_transform);
	m_pressure->setGridClass(openvdb::GridClass::GRID_FOG_VOLUME);
	m_pressure->setName("Pressure");

	m_rhs = openvdb::FloatGrid::create(float(0));
	m_rhs->setTransform(m_voxel_center_transform);
	m_rhs->setName("RHS");

	//velocity
	auto velocity = openvdb::Vec3fGrid::create(openvdb::Vec3f{ 0 });
	velocity->setTransform(m_voxel_center_transform);
	velocity->setName("Velocity");
	velocity->setGridClass(openvdb::GridClass::GRID_STAGGERED);

	m_packed_velocity.from_vec3(velocity,/*only topology*/true);

	m_packed_velocity_update = m_packed_velocity.deepCopy();
	m_packed_velocity_update.setName("Velocity_Update");

	m_packed_velocity_reflected = m_packed_velocity.deepCopy();
	m_packed_vel_after_p2g = m_packed_velocity.deepCopy();
	m_packed_velocity_for_pressure = m_packed_velocity.deepCopy();
	m_solved_viscosity_velocity = m_packed_velocity.deepCopy();

	//solid velocity
	m_solid_velocity = openvdb::Vec3fGrid::create(openvdb::Vec3f{ 0,0,0 });
	m_solid_velocity->setTransform(m_voxel_center_transform);
	m_solid_velocity->setGridClass(openvdb::GridClass::GRID_STAGGERED);
	m_solid_velocity->setName("Solid_Velocity");

	//face weights for pressure
	m_face_weight = openvdb::Vec3fGrid::create(openvdb::Vec3f{ 0,0,0 });
	m_face_weight->setName("Face_Weights");
	m_face_weight->setTransform(m_voxel_center_transform);
	m_face_weight->setGridClass(openvdb::GridClass::GRID_STAGGERED);

	//liquid sdf
	m_liquid_sdf = openvdb::FloatGrid::create(1.0f * m_dx);
	m_liquid_sdf->setGridClass(openvdb::GridClass::GRID_LEVEL_SET);
	m_liquid_sdf->setTransform(m_voxel_center_transform);
	m_liquid_sdf->setName("Liquid_SDF");

	//solid sdf
	//set background value to be positive to show it is away from the solid
	//treat it as the narrow band width
	m_solid_sdf = openvdb::FloatGrid::create(3.f * m_dx);
	m_solid_sdf->setTransform(m_voxel_vertex_transform);
	m_solid_sdf->setGridClass(openvdb::GridClass::GRID_LEVEL_SET);
	m_solid_sdf->setName("Solid_SDF");

	init_domain_solid_sdf();
}

void FLIP_Solver_OpenVDB::init_transforms()
{
	m_voxel_center_transform = openvdb::math::Transform::createLinearTransform(m_dx);
	m_voxel_vertex_transform = openvdb::math::Transform::createLinearTransform(m_dx);
	m_voxel_vertex_transform->postTranslate(openvdb::Vec3d{ -0.5,-0.5,-0.5 }*double(m_dx));

}

void FLIP_Solver_OpenVDB::init_domain_solid_sdf()
{
	auto halfvoxel = openvdb::Vec3f(0.5f * m_dx);
	auto theboxsdf = openvdb::tools::createLevelSetBox<openvdb::FloatGrid>(
		openvdb::BBoxd(m_domain_index_begin * m_dx - halfvoxel, m_domain_index_end * m_dx - halfvoxel), *m_voxel_vertex_transform);

	m_domain_solid_sdf = openvdb::FloatGrid::create(theboxsdf->background());
	for (auto iter = theboxsdf->tree().beginLeaf(); iter; ++iter) {
		m_domain_solid_sdf->tree().addLeaf(new openvdb::FloatTree::LeafNodeType(*iter));
	}

	auto make_container = [&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index) {
		for (auto iter = leaf.beginValueAll(); iter; ++iter) {
			if (iter.isValueOn()) {
				iter.setValue(-iter.getValue());
			}
			else {
				iter.setValue(theboxsdf->background());
			}
		}
	};
	auto sdfman = openvdb::tree::LeafManager<openvdb::FloatTree>(m_domain_solid_sdf->tree());
	sdfman.foreach(make_container);

	m_domain_solid_sdf->setName("Domain_SDF");
	m_domain_solid_sdf->setGridClass(openvdb::GridClass::GRID_LEVEL_SET);
	m_domain_solid_sdf->setTransform(m_voxel_vertex_transform);
}

void FLIP_Solver_OpenVDB::substep(FLIP_Object_OpenVDB& in_out_object, float substep_dt)
{
	switch (m_solver_parameters.simulation_method) {
	default:
	case SimulationMethod::ADVECTION_PROJECTION:
		if (m_solver_parameters.enable_viscosity == ViscosityOption::ON) {
			substep_advection_projection_with_viscosity(in_out_object, substep_dt);
		}
		else {
			substep_advection_projection(in_out_object, substep_dt);
		}
		break;
	case SimulationMethod::ADVECTION_REFLECITON:
		substep_advection_reflection_order1(in_out_object, substep_dt);
		break;
	}
}

void FLIP_Solver_OpenVDB::substep_advection_projection(FLIP_Object_OpenVDB& in_out_object, float substep_dt)
{
	m_substep_statistics.init();
	using duration_type = std::chrono::duration<double>;
	ZoneScoped;
	CSim::TimerMan::timer("Step").start();
	//set collision geometry
	CSim::TimerMan::timer("Step/update_solid_sdf").start();
	update_solid_sdf_vel(in_out_object.m_particles, m_collision_sdfvel);
	CSim::TimerMan::timer("Step/update_solid_sdf").stop();
	m_substep_statistics.update_solid_sdf_vel_time += duration_type(CSim::TimerMan::timer("Step/update_solid_sdf").lasttime());

	CSim::TimerMan::timer("Step/extrapolate").start();
	m_packed_velocity = in_out_object.m_velocity;
	//at least one extrapolation is need because the velocity after upgrade may not
	//contain all the 8 neighbor velocity sample around each particle
	extrapolate_velocity(3);
	CSim::TimerMan::timer("Step/extrapolate").stop();
	m_substep_statistics.extrapolate_time += duration_type(CSim::TimerMan::timer("Step/extrapolate").lasttime());

	CSim::TimerMan::timer("Step/adv").start();
	custom_move_points_and_set_flip_vel(in_out_object.m_particles,
		/*advector*/m_packed_velocity,
		/*carried information*/m_packed_velocity,
		/*vel of particle at grid*/m_packed_vel_after_p2g,
		/*collider*/m_solid_sdf,
		/*collider*/m_solid_velocity,
		m_solver_parameters.PIC_component, substep_dt,
		/*RK order*/ static_cast<int>(m_solver_parameters.particle_integration_method));
	CSim::TimerMan::timer("Step/adv").stop();
	m_substep_statistics.advection_time += duration_type(CSim::TimerMan::timer("Step/adv").lasttime());
	std::cout << "adv time: " << CSim::TimerMan::timer("Step/adv").lasttime() << std::endl;

	CSim::TimerMan::timer("Step/emit").start();
	for (auto& sdfvel : m_source_sdfvel) {
		emit_liquid(in_out_object.m_particles,
			sdfvel,
			m_solver_parameters.domain_min,
			m_solver_parameters.domain_max);
	}

	CSim::TimerMan::timer("Step/emit").stop();
	m_substep_statistics.emission_time += duration_type(CSim::TimerMan::timer("Step/emit").lasttime());
	
	CSim::TimerMan::timer("Step/sink").start();
	sink_liquid(in_out_object.m_particles, m_sink_sdfs);
	CSim::TimerMan::timer("Step/sink").stop();
	m_substep_statistics.sink_time += duration_type(CSim::TimerMan::timer("Step/sink").lasttime());

	//count particles
	for (auto iter = in_out_object.m_particles->tree().beginLeaf(); iter; ++iter) {
		m_substep_statistics.particle_count += iter->pointCount();
	}

	CSim::TimerMan::timer("Step/p2g").start();
	particle_to_grid(m_packed_velocity,
		m_liquid_sdf,
		in_out_object.m_particles,
		in_out_object.m_particle_radius);
	m_packed_vel_after_p2g = m_packed_velocity.deepCopy();
	CSim::TimerMan::timer("Step/p2g").stop();
	m_substep_statistics.p2g_time += duration_type(CSim::TimerMan::timer("Step/p2g").lasttime());
	std::cout << "p2g time: " << CSim::TimerMan::timer("Step/p2g").lasttime() << std::endl;

	//set simulation solid geometry for pressure solve
	CSim::TimerMan::timer("Step/update_solid_sdf").start();
	update_solid_sdf_vel(in_out_object.m_particles, m_solid_sdfvel);
	CSim::TimerMan::timer("Step/update_solid_sdf").stop();
	m_substep_statistics.update_solid_sdf_vel_time += duration_type(CSim::TimerMan::timer("Step/update_solid_sdf").lasttime());
	CSim::TimerMan::timer("Step/face_weight").start();
	calculate_face_weights();
	CSim::TimerMan::timer("Step/face_weight").stop();

	CSim::TimerMan::timer("Step/clamp_sdf").start();
	immerse_liquid_phi_in_solids();
	CSim::TimerMan::timer("Step/clamp_sdf").stop();

	CSim::TimerMan::timer("Step/extrapolate").start();
	//3 is necessary to cover all velocity sample needed for DOF
	extrapolate_velocity(3);
	CSim::TimerMan::timer("Step/extrapolate").stop();
	m_substep_statistics.extrapolate_time += duration_type(CSim::TimerMan::timer("Step/extrapolate").lasttime());


	CSim::TimerMan::timer("Step/bodyf").start();
	apply_body_force(substep_dt);
	CSim::TimerMan::timer("Step/bodyf").stop();
	//extrapolate_velocity(1);
	CSim::TimerMan::timer("Step/SIMD").start();
	solve_pressure_simd(substep_dt, /*reflection, twice the correction*/false);
	CSim::TimerMan::timer("Step/SIMD").stop();
	m_substep_statistics.poisson_time_total += duration_type(CSim::TimerMan::timer("Step/SIMD").lasttime());

	std::cout << "pressure time:" << CSim::TimerMan::timer("Step/SIMD").lasttime() << std::endl;
	in_out_object.m_velocity = m_packed_velocity;
	CSim::TimerMan::timer("Step").stop();
	m_substep_statistics.substep_total_time += duration_type(CSim::TimerMan::timer("Step").lasttime());
}

void FLIP_Solver_OpenVDB::substep_advection_reflection_order1(FLIP_Object_OpenVDB& in_out_object, float substep_dt)
{
	ZoneScoped;
	CSim::TimerMan::timer("Step").start();
	//split the sub_step into two steps
	float halfdt = substep_dt / 2;

	//A[f;v,dt]: advect field f with velocity v

	//first half step

	//set collision geometry
	CSim::TimerMan::timer("Step/update_solid_sdf").start();
	update_solid_sdf_vel(in_out_object.m_particles, m_collision_sdfvel);
	CSim::TimerMan::timer("Step/update_solid_sdf").stop();

	CSim::TimerMan::timer("Step/extrapolate").start();
	//at least one extrapolation is need because the velocity after upgrade may not
	//contain all the 8 neighbor velocity sample around each particle
	extrapolate_velocity(3);
	CSim::TimerMan::timer("Step/extrapolate").stop();

	CSim::TimerMan::timer("Step/adv").start();
	custom_move_points_and_set_flip_vel(in_out_object.m_particles,
		/*advector*/m_packed_velocity,
		/*carried information*/m_packed_velocity,
		/*vel of particle at grid*/m_packed_vel_after_p2g,
		/*collider*/m_solid_sdf,
		/*collider*/m_solid_velocity,
		m_solver_parameters.PIC_component, halfdt,
		/*RK order*/ static_cast<int>(m_solver_parameters.particle_integration_method));
	CSim::TimerMan::timer("Step/adv").stop();
	std::cout << "adv time: " << CSim::TimerMan::timer("Step/adv").lasttime() << std::endl;

	CSim::TimerMan::timer("Step/emit").start();
	for (auto& sdfvel : m_source_sdfvel) {
		emit_liquid(in_out_object.m_particles,
			sdfvel,
			m_solver_parameters.domain_min,
			m_solver_parameters.domain_max);
	}

	CSim::TimerMan::timer("Step/emit").stop();
	CSim::TimerMan::timer("Step/sink").start();
	sink_liquid(in_out_object.m_particles, m_sink_sdfs);
	CSim::TimerMan::timer("Step/sink").stop();

	CSim::TimerMan::timer("Step/p2g").start();
	particle_to_grid(m_packed_velocity,
		m_liquid_sdf,
		in_out_object.m_particles,
		in_out_object.m_particle_radius);
	m_packed_vel_after_p2g = m_packed_velocity.deepCopy();
	CSim::TimerMan::timer("Step/p2g").stop();
	std::cout << "p2g time: " << CSim::TimerMan::timer("Step/p2g").lasttime() << std::endl;


	//set simulation solid geometry for pressure solve
	CSim::TimerMan::timer("Step/update_solid_sdf").start();
	update_solid_sdf_vel(in_out_object.m_particles, m_solid_sdfvel);
	CSim::TimerMan::timer("Step/update_solid_sdf").stop();
	CSim::TimerMan::timer("Step/face_weight").start();
	calculate_face_weights();
	CSim::TimerMan::timer("Step/face_weight").stop();

	CSim::TimerMan::timer("Step/clamp_sdf").start();
	immerse_liquid_phi_in_solids();
	CSim::TimerMan::timer("Step/clamp_sdf").stop();

	CSim::TimerMan::timer("Step/extrapolate").start();
	//3 is necessary to cover all velocity sample needed for DOF
	extrapolate_velocity(3);
	CSim::TimerMan::timer("Step/extrapolate").stop();

	CSim::TimerMan::timer("Step/bodyf").start();
	apply_body_force(halfdt);
	CSim::TimerMan::timer("Step/bodyf").stop();

	CSim::TimerMan::timer("Step/SIMD").start();
	solve_pressure_simd(halfdt, /*reflection, twice the correction*/true);
	CSim::TimerMan::timer("Step/SIMD").stop();

	//now the m_velocity is the divergence-free velocity, and we have an m_velocity_reflected
	auto u_half = m_packed_velocity;

	/*----------------4--------------*/

	//set collision geometry
	CSim::TimerMan::timer("Step/update_solid_sdf").start();
	update_solid_sdf_vel(in_out_object.m_particles, m_collision_sdfvel);
	CSim::TimerMan::timer("Step/update_solid_sdf").stop();

	//u_half (m_velocity) is used for advection, so it needs to be extrapolated
	//the u0 may not be able to cover the extra space due to first advection
	//also need to be extrapolated 
	//vdb_velocity_extrapolator::extrapolate(4, m_velocity);
	//vdb_velocity_extrapolator::extrapolate(2, m_velocity_reflected);
	CSim::TimerMan::timer("Step/extrapolate").start();
	vdb_velocity_extrapolator::union_extrapolate(3, m_packed_velocity.v[0], m_packed_velocity.v[1], m_packed_velocity.v[2], &(m_liquid_sdf->tree()));
	vdb_velocity_extrapolator::union_extrapolate(1, m_packed_velocity_reflected.v[0], m_packed_velocity_reflected.v[1], m_packed_velocity_reflected.v[2], &(m_liquid_sdf->tree()));
	CSim::TimerMan::timer("Step/extrapolate").stop();

	CSim::TimerMan::timer("Step/adv").start();
	custom_move_points_and_set_flip_vel(in_out_object.m_particles,
		/*advector*/m_packed_velocity,
		/*carried information*/m_packed_velocity_reflected,
		/*vel of particle at grid*/m_packed_vel_after_p2g,
		/*collider*/m_solid_sdf,
		/*collider*/m_solid_velocity,
		m_solver_parameters.PIC_component, halfdt,
		/*RK order*/ static_cast<int>(m_solver_parameters.particle_integration_method));
	CSim::TimerMan::timer("Step/adv").stop();

	CSim::TimerMan::timer("Step/emit").start();
	for (auto& sdfvel : m_source_sdfvel) {
		emit_liquid(in_out_object.m_particles,
			sdfvel,
			m_solver_parameters.domain_min,
			m_solver_parameters.domain_max);
	}

	CSim::TimerMan::timer("Step/emit").stop();
	CSim::TimerMan::timer("Step/sink").start();
	sink_liquid(in_out_object.m_particles, m_sink_sdfs);
	CSim::TimerMan::timer("Step/sink").stop();

	CSim::TimerMan::timer("Step/p2g").start();
	particle_to_grid(m_packed_velocity,
		m_liquid_sdf,
		in_out_object.m_particles,
		in_out_object.m_particle_radius);
	m_packed_vel_after_p2g = m_packed_velocity.deepCopy();
	CSim::TimerMan::timer("Step/p2g").stop();
	std::cout << "p2g time: " << CSim::TimerMan::timer("Step/p2g").lasttime() << std::endl;

	//set simulation solid geometry for pressure solve
	CSim::TimerMan::timer("Step/update_solid_sdf").start();
	update_solid_sdf_vel(in_out_object.m_particles, m_solid_sdfvel);
	CSim::TimerMan::timer("Step/update_solid_sdf").stop();
	CSim::TimerMan::timer("Step/face_weight").start();
	calculate_face_weights();
	CSim::TimerMan::timer("Step/face_weight").stop();

	CSim::TimerMan::timer("Step/clamp_sdf").start();
	immerse_liquid_phi_in_solids();
	CSim::TimerMan::timer("Step/clamp_sdf").stop();

	CSim::TimerMan::timer("Step/extrapolate").start();
	//3 is necessary to cover all velocity sample needed for DOF
	extrapolate_velocity(3);
	CSim::TimerMan::timer("Step/extrapolate").stop();

	CSim::TimerMan::timer("Step/bodyf").start();
	apply_body_force(halfdt);
	CSim::TimerMan::timer("Step/bodyf").stop();
	//extrapolate_velocity(1);
	CSim::TimerMan::timer("Step/SIMD").start();
	solve_pressure_simd(halfdt, /*reflection, twice the correction*/false);
	CSim::TimerMan::timer("Step/SIMD").stop();
	std::cout << "pressure time:" << CSim::TimerMan::timer("Step/SIMD").lasttime() << std::endl;
	CSim::TimerMan::timer("Step").stop();
}

void FLIP_Solver_OpenVDB::substep_advection_projection_with_viscosity(FLIP_Object_OpenVDB& in_out_object, float substep_dt)
{
	m_substep_statistics.init();
	using duration_type = std::chrono::duration<double>;
	CSim::TimerMan::timer("Step").start();

	//set collision geometry
	CSim::TimerMan::timer("Step/update_solid_sdf").start();
	update_solid_sdf_vel(in_out_object.m_particles, m_collision_sdfvel);
	CSim::TimerMan::timer("Step/update_solid_sdf").stop();
	m_substep_statistics.update_solid_sdf_vel_time += duration_type(CSim::TimerMan::timer("Step/update_solid_sdf").lasttime());

	CSim::TimerMan::timer("Step/extrapolate").start();
	//at least one extrapolation is need because the velocity after upgrade may not
	//contain all the 8 neighbor velocity sample around each particle
	extrapolate_velocity(3);
	CSim::TimerMan::timer("Step/extrapolate").stop();
	m_substep_statistics.extrapolate_time += duration_type(CSim::TimerMan::timer("Step/extrapolate").lasttime());

	CSim::TimerMan::timer("Step/adv").start();
	custom_move_points_and_set_flip_vel(in_out_object.m_particles,
		/*advector*/m_packed_velocity,
		/*carried information*/m_packed_velocity,
		/*vel of particle at grid*/m_packed_vel_after_p2g,
		/*collider*/m_solid_sdf,
		/*collider*/m_solid_velocity,
		m_solver_parameters.PIC_component, substep_dt,
		/*RK order*/ static_cast<int>(m_solver_parameters.particle_integration_method));
	CSim::TimerMan::timer("Step/adv").stop();
	m_substep_statistics.advection_time += duration_type(CSim::TimerMan::timer("Step/adv").lasttime());
	std::cout << "adv time: " << CSim::TimerMan::timer("Step/adv").lasttime() << std::endl;

	CSim::TimerMan::timer("Step/emit").start();
	for (auto& sdfvel : m_source_sdfvel) {
		emit_liquid(in_out_object.m_particles,
			sdfvel,
			m_solver_parameters.domain_min,
			m_solver_parameters.domain_max);
	}
	CSim::TimerMan::timer("Step/emit").stop();
	m_substep_statistics.emission_time += duration_type(CSim::TimerMan::timer("Step/emit").lasttime());
	CSim::TimerMan::timer("Step/sink").start();
	sink_liquid(in_out_object.m_particles, m_sink_sdfs);
	CSim::TimerMan::timer("Step/sink").stop();
	m_substep_statistics.sink_time += duration_type(CSim::TimerMan::timer("Step/sink").lasttime());
	for (auto iter = in_out_object.m_particles->tree().beginLeaf(); iter; ++iter) {
		m_substep_statistics.particle_count += iter->pointCount();
	}

	CSim::TimerMan::timer("Step/p2g").start();
	particle_to_grid(m_packed_velocity,
		m_liquid_sdf,
		in_out_object.m_particles,
		in_out_object.m_particle_radius);
	m_packed_vel_after_p2g = m_packed_velocity.deepCopy();
	CSim::TimerMan::timer("Step/p2g").stop();
	m_substep_statistics.p2g_time += duration_type(CSim::TimerMan::timer("Step/p2g").lasttime());
	std::cout << "p2g time: " << CSim::TimerMan::timer("Step/p2g").lasttime() << std::endl;

	//set simulation solid geometry for pressure solve
	CSim::TimerMan::timer("Step/update_solid_sdf").start();
	update_solid_sdf_vel(in_out_object.m_particles, m_solid_sdfvel);
	CSim::TimerMan::timer("Step/update_solid_sdf").stop();
	m_substep_statistics.update_solid_sdf_vel_time += duration_type(CSim::TimerMan::timer("Step/update_solid_sdf").lasttime());

	CSim::TimerMan::timer("Step/face_weight").start();
	calculate_face_weights();
	CSim::TimerMan::timer("Step/face_weight").stop();

	CSim::TimerMan::timer("Step/clamp_sdf").start();
	immerse_liquid_phi_in_solids();
	CSim::TimerMan::timer("Step/clamp_sdf").stop();

	CSim::TimerMan::timer("Step/bodyf").start();
	apply_body_force(substep_dt);
	CSim::TimerMan::timer("Step/bodyf").stop();

	CSim::TimerMan::timer("Step/extrapolate").start();
	//3 is necessary to cover all velocity sample needed for DOF
	extrapolate_velocity(3);
	CSim::TimerMan::timer("Step/extrapolate").stop();
	m_substep_statistics.extrapolate_time += duration_type(CSim::TimerMan::timer("Step/extrapolate").lasttime());

	CSim::TimerMan::timer("Step/SIMD").start();
	solve_pressure_simd(substep_dt, /*reflection, twice the correction*/false);
	CSim::TimerMan::timer("Step/SIMD").stop();
	std::cout << "pressure time:" << CSim::TimerMan::timer("Step/SIMD").lasttime() << std::endl;
	m_substep_statistics.poisson_time_total += duration_type(CSim::TimerMan::timer("Step/SIMD").lasttime());

	CSim::TimerMan::timer("Step/viscosity").start();
	solve_viscosity(in_out_object, substep_dt);
	CSim::TimerMan::timer("Step/viscosity").stop();
	m_substep_statistics.viscosity_time_total = duration_type(CSim::TimerMan::timer("Step/viscosity").lasttime());

	CSim::TimerMan::timer("Step/extrapolate").start();
	//3 is necessary to cover all velocity sample needed for DOF
	extrapolate_velocity(3);
	CSim::TimerMan::timer("Step/extrapolate").stop();
	m_substep_statistics.extrapolate_time += duration_type(CSim::TimerMan::timer("Step/extrapolate").lasttime());

	CSim::TimerMan::timer("Step/SIMD").start();
	solve_pressure_simd(substep_dt, /*reflection, twice the correction*/false);
	CSim::TimerMan::timer("Step/SIMD").stop();
	m_substep_statistics.poisson_time_total += duration_type(CSim::TimerMan::timer("Step/SIMD").lasttime());

	std::cout << "pressure time:" << CSim::TimerMan::timer("Step/SIMD").lasttime() << std::endl;
	CSim::TimerMan::timer("Step").stop();
	m_substep_statistics.substep_total_time += duration_type(CSim::TimerMan::timer("Step").lasttime());
}

void FLIP_Solver_OpenVDB::apply_body_force(float in_dt)
{
	ZoneScoped;
	openvdb::Vec3f body_f = m_solver_parameters.gravity;

	for (int channel = 0; channel < 3; channel++) {
		auto add_body_force_op = [&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index leafpos) {
			for (auto iter = leaf.beginValueOn(); iter; ++iter) {
				iter.setValue(iter.getValue() + body_f[channel] * in_dt);
			}
		};//end velocity_update_op

		auto vel_leafman = openvdb::tree::LeafManager<openvdb::FloatTree>(m_packed_velocity.v[channel]->tree());
		vel_leafman.foreach(add_body_force_op);
	}
}

void FLIP_Solver_OpenVDB::apply_pressure_gradient(float in_dt, bool reflection)
{

	m_packed_velocity_update = m_packed_velocity.deepCopy();
	m_packed_velocity_update.setName("Velocity_Update");

	if (reflection) {
		m_packed_velocity_reflected = m_packed_velocity.deepCopy();
		m_packed_velocity_reflected.setName("velocity_uhalf");
	}

	const float boundary_friction_coef = m_solver_parameters.boundary_friction_coefficient;

	//CSim::TimerMan::timer("Step/SIMD/grad/apply").start();
	for (int channel = 0; channel < 3; channel++) {
		auto channel_update_op = [&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index leafpos) {
			auto phi_axr{ m_liquid_sdf->getConstUnsafeAccessor() };
			auto weight_axr{ m_face_weight->getConstUnsafeAccessor() };
			auto solid_vel_axr{ m_solid_velocity->getConstUnsafeAccessor() };
			auto pressure_axr{ m_pressure->getConstUnsafeAccessor() };
			openvdb::FloatTree::LeafNodeType* reflected_channel_leaf;
			if (reflection) {
				reflected_channel_leaf = m_packed_velocity_reflected.v[channel]->tree().probeLeaf(leaf.origin());
			}
			auto update_channel_leaf = m_packed_velocity_update.v[channel]->tree().probeLeaf(leaf.origin());

			for (auto iter = leaf.beginValueOn(); iter; ++iter) {
				bool has_update = false;
				const auto gcoord = iter.getCoord();
				auto updated_vel = iter.getValue();
				auto reflected_vel = updated_vel;
				float vel_update = 0.0f;
				auto face_weight = weight_axr.getValue(gcoord)[channel];

				auto lower_gcoord = gcoord;
				lower_gcoord[channel] -= 1;

				if (face_weight > 0.0f) {
					//this face has liquid component
					//does it have any dof on its side?

					bool has_pressure = pressure_axr.isValueOn(gcoord);
					bool has_pressure_below = pressure_axr.isValueOn(lower_gcoord);

					if (has_pressure || has_pressure_below) {
						has_update = true;
					}

					if (has_update) {
						auto phi_this = phi_axr.getValue(gcoord);
						auto phi_below = phi_axr.getValue(lower_gcoord);
						float p_this = pressure_axr.getValue(gcoord);
						float p_below = pressure_axr.getValue(lower_gcoord);

						float theta = 1.0f;
						if (phi_this >= 0 || phi_below >= 0) {
							theta = fraction_inside(phi_below, phi_this);
							if (theta < 0.02f) theta = 0.02f;
						}

						vel_update = -in_dt * (float)(p_this - p_below) / m_dx / theta;
						updated_vel += vel_update;
						reflected_vel = updated_vel + vel_update;

						if (face_weight < 1.0f) {
							auto solid_vel = solid_vel_axr.getValue(gcoord)[channel];
							//mix the solid velocity and fluid velocity if friction is expected
							float solid_fraction = (1.0f - face_weight) * boundary_friction_coef;
							updated_vel = (1.0f - solid_fraction) * updated_vel + (solid_fraction)*solid_vel;
						}
					}//end if any dofs on two sides
				}//if this face has positive face weight

				if (has_update) {
					iter.setValue(updated_vel);
					if (reflection) {
						reflected_channel_leaf->setValueOnly(iter.offset(), reflected_vel);
					}

					update_channel_leaf->setValueOnly(iter.offset(), vel_update);
				}
				else {
					iter.setValueOff();
					if (reflection) {
						reflected_channel_leaf->setValueOff(iter.offset());
					}
					update_channel_leaf->setValueOff(iter.offset());
				}
			}//loop over all velocity faces
		};//end channel_update_op

		auto vel_leafman = openvdb::tree::LeafManager<openvdb::FloatTree>(m_packed_velocity.v[channel]->tree());
		vel_leafman.foreach(channel_update_op);
	}//end for each channel
	//CSim::TimerMan::timer("Step/SIMD/grad/apply").stop();

}

void FLIP_Solver_OpenVDB::calculate_face_weights()
{
	ZoneScoped;
	m_face_weight = openvdb::Vec3fGrid::create(openvdb::Vec3f{ 1.0f });
	m_face_weight->setTree(std::make_shared<openvdb::Vec3fTree>(
		m_liquid_sdf->tree(), openvdb::Vec3f{ 1.0f }, openvdb::TopologyCopy()));
	openvdb::tools::dilateActiveValues(m_face_weight->tree(), 2, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);
	m_face_weight->setName("Face_Weights");
	m_face_weight->setTransform(m_liquid_sdf->transformPtr());
	m_face_weight->setGridClass(openvdb::GridClass::GRID_STAGGERED);
	auto set_face_weight_op = [&](openvdb::Vec3fTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		auto solid_axr{ m_solid_sdf->getConstUnsafeAccessor() };
		//solid sdf
		float ssdf[2][2][2];
		openvdb::Vec3f uvwweight{ 1.f };
		for (auto offset = 0; offset < leaf.SIZE; offset++) {
			if (leaf.isValueMaskOff(offset)) {
				continue;
			}
			for (int ii = 0; ii < 2; ii++) {
				for (int jj = 0; jj < 2; jj++) {
					for (int kk = 0; kk < 2; kk++) {
						ssdf[ii][jj][kk] = solid_axr.getValue(leaf.offsetToGlobalCoord(offset) + openvdb::Coord{ ii,jj,kk });
					}
				}
			}//end retrieve eight solid sdfs on the voxel corners

			//fraction_inside(bl,br,tl,tr)
			//tl-----tr
			//|      |
			//|      |
			//bl-----br

			//look from positive x direction

			//   ^z
			//(0,0,1)------(0,1,1)
			//   |            |
			//(0,0,0)------(0,1,0)>y
			//uweight
			uvwweight[0] = 1.0f - fraction_inside(
				ssdf[0][0][0],
				ssdf[0][1][0],
				ssdf[0][0][1],
				ssdf[0][1][1]);
			uvwweight[0] = std::max(0.f, std::min(uvwweight[0], 1.f));

			//look from positive y direction
			//   ^x              
			//(1,0,0)------(1,0,1)
			//   |            |
			//(0,0,0)------(0,0,1)>z
			//vweight
			uvwweight[1] = 1.0f - fraction_inside(
				ssdf[0][0][0],
				ssdf[0][0][1],
				ssdf[1][0][0],
				ssdf[1][0][1]);
			uvwweight[1] = std::max(0.f, std::min(uvwweight[1], 1.f));

			//look from positive z direction
			//   ^y              
			//(0,1,0)------(1,1,0)
			//   |            |
			//(0,0,0)------(1,0,0)>x
			//wweight
			uvwweight[2] = 1.0f - fraction_inside(
				ssdf[0][0][0],
				ssdf[1][0][0],
				ssdf[0][1][0],
				ssdf[1][1][0]);
			uvwweight[2] = std::max(0.f, std::min(uvwweight[2], 1.f));
			for (int c = 0; c < 3; c++) {
				if (uvwweight[c] < 1e-2f) {
					uvwweight[c] = 0.f;
				}
			}
			leaf.setValueOn(offset, uvwweight);
		}//end for all offset
	};//end set face weight op

	auto leafman = openvdb::tree::LeafManager<openvdb::Vec3fTree>(m_face_weight->tree());
	leafman.foreach(set_face_weight_op);
}

void FLIP_Solver_OpenVDB::extrapolate_velocity(int layer)
{
	ZoneScoped;
	vdb_velocity_extrapolator::union_extrapolate(layer,
		m_packed_velocity.v[0],
		m_packed_velocity.v[1],
		m_packed_velocity.v[2],
		&(m_liquid_sdf->tree()));
}

float FLIP_Solver_OpenVDB::cfl()
{
	float total_max_v = 0.f;
	float total_top_v = 0.f;
	for (int i = 0; i < 3; i++) {
		std::vector<float> max_v_per_leaf;
		max_v_per_leaf.assign(m_packed_velocity.v[i]->tree().leafCount(), 0);

		auto find_per_leaf_max_vel = [&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index leafpos) {
			float max_v = 0;
			for (auto iter = leaf.beginValueOn(); iter; ++iter) {
				if (std::isnan(iter.getValue())) {
					printf("velocity nan\n");
					exit(-1);
				}
				if (max_v < std::abs(iter.getValue())) {
					max_v = std::abs(iter.getValue());
				}
			}
			max_v_per_leaf[leafpos] = max_v;
		};// end find per leaf max vel

		auto leafman = openvdb::tree::LeafManager <openvdb::FloatTree>(m_packed_velocity.v[i]->tree());
		leafman.foreach(find_per_leaf_max_vel);

		float max_v = 0;
		if (max_v_per_leaf.empty()) {
			max_v = std::abs(m_packed_velocity.v[i]->background());
			total_max_v = std::max(max_v, total_max_v);
			total_top_v = std::max(max_v, total_top_v);
		}
		else {
			//dont take the maximum number
		//take the 90%?
			int nleaf = max_v_per_leaf.size();
			int top_percentile = nleaf * 99 / 100 - 1;
			top_percentile = std::max(top_percentile, 0);
			std::nth_element(max_v_per_leaf.begin(), max_v_per_leaf.begin() + nleaf - 1, max_v_per_leaf.end());
			std::nth_element(max_v_per_leaf.begin(), max_v_per_leaf.begin() + top_percentile, max_v_per_leaf.end());

			max_v = max_v_per_leaf.back();
			total_max_v = std::max(max_v, total_max_v);
			max_v = max_v_per_leaf[top_percentile];
			total_top_v = std::max(max_v, total_top_v);
			max_v = max_v_per_leaf[0];
		}
	}

	printf("max velocity component:%f\n", total_max_v);
	printf("cfl velocity:%f\n", total_top_v);
	return m_dx / (std::abs(total_max_v) + 1e-6f);
}
float FLIP_Solver_OpenVDB::particle_cfl(openvdb::points::PointDataGrid::Ptr in_particles)
{
	std::vector<float> per_leaf_max_vel;
	auto nleaf = in_particles->tree().leafCount();
	if (nleaf == 0) {
		return 1.0f;
	}

	per_leaf_max_vel.assign(nleaf, 0.f);
	auto per_leaf_max_vel_reader = [&](openvdb::points::PointDataTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		//reader
		// Extract the velocity attribute from the leaf by name (v is velocity).
		const openvdb::points::AttributeArray& velocityArray =
			leaf.attributeArray("v");
		auto v_handle_ptr = openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_Object_OpenVDB::VelocityCodec>::create(velocityArray);

		float max_vel = 0.f;
		auto nparticles = leaf.getLastValue();
		for (int i = 0; i < nparticles;++i) {
			auto v = v_handle_ptr->get(i);
			max_vel = std::max(max_vel, std::abs(v[0]));
			max_vel = std::max(max_vel, std::abs(v[1]));
			max_vel = std::max(max_vel, std::abs(v[2]));
		}
		per_leaf_max_vel[leafpos] = max_vel;
	};
	
	openvdb::tree::LeafManager<openvdb::points::PointDataTree> leafman(in_particles->tree());

	leafman.foreach(per_leaf_max_vel_reader);

	float max_vel = per_leaf_max_vel[0];
	for (int i = 0; i < nleaf; ++i) {
		max_vel = std::max(max_vel, per_leaf_max_vel[i]);
	}
	float dx = in_particles->transform().voxelSize()[0];
	printf("particle max v:%f\n", max_vel);
	return dx / (max_vel + 1e-6);
}

void FLIP_Solver_OpenVDB::immerse_liquid_phi_in_solids()
{
	ZoneScoped;

	auto correct_liquid_phi_in_solid = [&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		//detech if there is solid
		if (m_solid_sdf->tree().probeConstLeaf(leaf.origin())) {
			auto const_solid_axr{ m_solid_sdf->getConstUnsafeAccessor() };
			auto shift{ openvdb::Vec3R{0.5} };
			for (auto iter = leaf.beginValueOn(); iter; ++iter) {
				auto voxel_solid_sdf = openvdb::tools::BoxSampler::sample(
					const_solid_axr, iter.getCoord() + shift);
				if (voxel_solid_sdf < 0) {
					//leaf.setValueOn(offset, -voxel_solid_sdf);
					iter.setValue(iter.getValue() - 0.5f * m_dx);
				}
			}//end for all voxel
		}//end there is solid in this leaf
	};//end correct_liquid_phi_in_solid

	auto phi_manager = openvdb::tree::LeafManager<openvdb::FloatTree>(m_liquid_sdf->tree());
	phi_manager.foreach(correct_liquid_phi_in_solid);

	auto ref_liquid_sdf = m_liquid_sdf->deepCopy();

	openvdb::tools::dilateActiveValues(m_liquid_sdf->tree(), 1, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);
	//this is to be called by a solid manager
	auto immerse_liquid_into_solid = [&](openvdb::FloatTree::LeafNodeType& leaf, openvdb::Index leafpos) {
		//detech if there is solid
		if (m_solid_sdf->tree().probeConstLeaf(leaf.origin())) {
			auto reference_liquid_sdf_axr{ ref_liquid_sdf->getConstUnsafeAccessor() };
			auto const_solid_axr{ m_solid_sdf->getConstUnsafeAccessor() };
			auto shift{ openvdb::Vec3R{0.5} };
			auto solid_weight_axr{ m_face_weight->getConstUnsafeAccessor() };

			for (auto iter = leaf.beginValueOn(); iter; ++iter) {
				auto voxel_solid_sdf = openvdb::tools::BoxSampler::sample(
					m_solid_sdf->tree(), leaf.offsetToGlobalCoord(iter.offset()).asVec3d() + shift);

				if (voxel_solid_sdf < 0) {
					bool found_liquid_neib = false;
					float min_fluid_sdf = m_dx * 3.0f;
					for (int i_neib = 0; i_neib < 6 && !found_liquid_neib; i_neib++) {
						int component = i_neib / 2;
						int positive_dir = (i_neib % 2 == 0);

						auto test_coord = iter.getCoord();
						auto weight_coord = iter.getCoord();
						if (positive_dir) {
							test_coord[component]++;
							weight_coord[component]++;
						}
						else {
							test_coord[component]--;
						}
						bool can_influence_neighbor = true;// solid_weight_axr.getValue(weight_coord)[component] > 0;
						min_fluid_sdf = std::min(min_fluid_sdf, reference_liquid_sdf_axr.getValue(test_coord));
						found_liquid_neib |= (reference_liquid_sdf_axr.getValue(test_coord) < 0) && can_influence_neighbor;
					}//end for 6 direction


					//mark this voxel as liquid
					if (found_liquid_neib) {
						float current_sdf = iter.getValue();
						//iter.setValue(current_sdf - mDx * 0.5f);
						iter.setValue(min_fluid_sdf);
						//iter.setValue(-mDx * 0.01);
						//iter.setValueOn();
					}
					else {
						//in solid, but found no liquid neighbors, or is surrounded by 6 solid faces
						//becoming an isolated DOF
						if (iter.getValue() < 0) {
							iter.setValue(std::max(m_liquid_sdf->background(), -voxel_solid_sdf));
						}
					}
				}//end if this voxel is inside solid

			}//end for all voxel
		}//end there is solid in this leaf
	};//end immerse_liquid_into_solid

	phi_manager.foreach(immerse_liquid_into_solid);

	//dilate the liquid sdf just to indicate where the velocity extrapolation should go
	openvdb::tools::dilateActiveValues(m_liquid_sdf->tree(), 1, openvdb::tools::NearestNeighbors::NN_FACE, openvdb::tools::TilePolicy::EXPAND_TILES);
}

//remove all particles inside solids, in sink zone, and outside of the bounding box
void FLIP_Solver_OpenVDB::sink_liquid(openvdb::points::PointDataGrid::Ptr in_out_particles, const std::vector<openvdb::FloatGrid::Ptr>& in_sdfs)
{
	ZoneScoped;

	auto partman = openvdb::tree::LeafManager<openvdb::points::PointDataTree>(in_out_particles->tree());

	std::vector<openvdb::points::PointDataTree::LeafNodeType*> leaves;
	leaves.reserve(partman.leafCount());
	partman.getNodes(leaves);

	openvdb::Vec3d idx_domain_min = m_domain_index_begin - openvdb::Vec3d{ 0.5 },
		idx_domain_max = m_domain_index_end - openvdb::Vec3d{ 0.5 };

	auto pnamepair = PositionAttribute::attributeType();
	auto position_attribute_descriptor = openvdb::points::AttributeSet::Descriptor::create(pnamepair);

	auto vnamepair = VelocityAttribute::attributeType();
	auto pv_attribute_descriptor = position_attribute_descriptor->duplicateAppend("v", vnamepair);

	auto sink_leaf = [&](const tbb::blocked_range<size_t>& r) {
		std::vector<openvdb::tree::ValueAccessor<const openvdb::FloatTree>> sink_axrs;
		for (size_t isink = 0; isink < in_sdfs.size(); isink++) {
			sink_axrs.push_back(openvdb::tree::ValueAccessor<const openvdb::FloatTree>{ in_sdfs[isink]->tree() });
		}

		auto particle_voxel_inside_domain = [&](const openvdb::Coord& voxel_coord) {
			return (voxel_coord[0] >= m_domain_index_begin[0] &&
				voxel_coord[1] >= m_domain_index_begin[1] &&
				voxel_coord[2] >= m_domain_index_begin[2] &&
				voxel_coord[0] < m_domain_index_end[0] &&
				voxel_coord[1] < m_domain_index_end[1] &&
				voxel_coord[2] < m_domain_index_end[2]);
		};

		auto touch_sink = [&](const openvdb::Vec3d& pipos, float delta = 0) {
			//check domain boundary in index space
			if (pipos[0] <= idx_domain_min[0] || pipos[0] >= idx_domain_max[0]
				|| pipos[1] <= idx_domain_min[1] || pipos[1] >= idx_domain_max[1]
				|| pipos[2] <= idx_domain_min[2] || pipos[2] >= idx_domain_max[2]) {
				return true;
			}

			openvdb::Vec3d wpos = in_out_particles->indexToWorld(pipos);

			//check sinks
			for (int isink = 0; isink < in_sdfs.size(); isink++) {
				openvdb::Vec3d sinkipos = in_sdfs[isink]->worldToIndex(wpos);
				if (openvdb::tools::BoxSampler::sample(sink_axrs[isink], sinkipos) < delta) {
					return true;
				}
			}

			return false;
		};//end touch_sink

		for (auto leafpos = r.begin(); leafpos != r.end(); leafpos++) {
			auto leaforigin = leaves[leafpos]->origin();

			bool corner_sink = false;
			//anyone of its corners touch sink zone then it is worth processing
			for (int i = 0; i <= 8 && !corner_sink; i += 4) {
				for (int j = 0; j <= 8 && !corner_sink; j += 4) {
					for (int k = 0; k <= 8 && !corner_sink; k += 4) {
						corner_sink |= touch_sink(leaforigin.offset(i, j, k).asVec3d() - openvdb::Vec3d{ 0.5 });
					}//k
				}//j
			}//i

			if (!corner_sink) { continue; }
			//below this line, at least one corner of the particle leaf touches sink zone
			//loop over all particles and remove those in sink zone

			//this bulk is empty so skip the deletion
			if (leaves[leafpos]->getLastValue() == 0) {
				continue;
			}

			//new p v arrays
			custom_vector_type<openvdb::Vec3f> new_p, new_v;
			std::vector<openvdb::PointDataIndex32> new_idxoffset;
			new_idxoffset.reserve(leaves[leafpos]->SIZE);

			//pv readers
			const openvdb::points::AttributeArray& positionArray =
				leaves[leafpos]->attributeArray("P");
			// Extract the velocity attribute from the leaf by name (v is velocity).
			const openvdb::points::AttributeArray& velocityArray =
				leaves[leafpos]->attributeArray("v");

			// Create read handles for position and velocity
			openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_Object_OpenVDB::PositionCodec> positionHandle(positionArray);
			openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_Object_OpenVDB::VelocityCodec> velocityHandle(velocityArray);


			int emitted_particles = 0;
			//relaxed criteria for selecting voxels
			//allow further criteria for particles
			float voxeldelta = 1.0f * in_out_particles->transform().voxelSize()[0];

			//loop over all voxels
			for (auto offset = 0; offset < leaves[leafpos]->SIZE; ++offset) {
				int idxend = leaves[leafpos]->getValue(offset);
				int idxbegin = 0;
				if (offset != 0) {
					idxbegin = leaves[leafpos]->getValue(offset - 1);
				}

				if (idxend == idxbegin) {
					new_idxoffset.push_back(emitted_particles);
					continue;
				}

				auto voxelipos = leaves[leafpos]->offsetToGlobalCoord(offset);

				if (!particle_voxel_inside_domain(voxelipos)) {
					new_idxoffset.push_back(emitted_particles);
					continue;
				}

				if (touch_sink(voxelipos.asVec3d(), voxeldelta)) {
					//emit those are qualified
					for (auto idx = idxbegin; idx < idxend; idx++) {
						auto particleipos = positionHandle.get(idx) + voxelipos;
						if (!touch_sink(particleipos)) {
							new_p.push_back(positionHandle.get(idx));
							new_v.push_back(velocityHandle.get(idx));
							emitted_particles++;
						}
					}
				}
				else {
					//emit as usual
					for (auto idx = idxbegin; idx < idxend; idx++) {
						new_p.push_back(positionHandle.get(idx));
						new_v.push_back(velocityHandle.get(idx));
						emitted_particles++;
					}
				}
				new_idxoffset.push_back(emitted_particles);
			}//end for all voxels

			//assign new attributes
			if (new_p.empty()) {
				leaves[leafpos]->clearAttributes();
			}
			else {
				auto local_pv_descriptor = pv_attribute_descriptor;
				leaves[leafpos]->initializeAttributes(position_attribute_descriptor, new_p.size());
				leaves[leafpos]->appendAttribute(leaves[leafpos]->attributeSet().descriptor(), local_pv_descriptor, 1);
				leaves[leafpos]->setOffsets(new_idxoffset);

				//set the positions and velocities
				//get the new attribute arrays
				openvdb::points::AttributeArray& posarray = leaves[leafpos]->attributeArray("P");
				openvdb::points::AttributeArray& varray = leaves[leafpos]->attributeArray("v");

				// Create read handles for position and velocity
				openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_Object_OpenVDB::PositionCodec> posWHandle(posarray);
				openvdb::points::AttributeWriteHandle<openvdb::Vec3f, FLIP_Object_OpenVDB::VelocityCodec> vWHandle(varray);

				for (auto iter = leaves[leafpos]->beginIndexOn(); iter; ++iter) {
					posWHandle.set(*iter, new_p[*iter]);
					vWHandle.set(*iter, new_v[*iter]);
				}
			}//end if has points to write to attribute
		}//end for ranged leaf pos
	};//end sink_leaf lambda

	tbb::parallel_for(tbb::blocked_range<size_t>(0, leaves.size()), sink_leaf);

	in_out_particles->pruneGrid();
}

void FLIP_Solver_OpenVDB::solve_pressure_simd(float in_dt, bool reflection)
{
	using duration_type = std::chrono::duration<double>;
	ZoneScoped;
	//skip if there is no dof to solve
	if (m_liquid_sdf->tree().leafCount() == 0) {
		return;
	}
	CSim::TimerMan::timer("Step/SIMD/build").start();
	m_packed_velocity_for_pressure = m_packed_velocity.deepCopy();
	m_packed_velocity_for_pressure.setName("m_velocity_before_pressure_solve");
	//has leaf but empty leaf
	//test construct levels

	auto lhs_matrix = simd_uaamg::LaplacianWithLevel::
		createPressurePoissonLaplacian(m_liquid_sdf, m_face_weight, in_dt);
	auto simd_solver = simd_uaamg::PoissonSolver(lhs_matrix);
	simd_solver.mRelativeTolerance = m_solver_parameters.pressure_solver_accuracy;
	simd_solver.mSmoother = simd_uaamg::PoissonSolver::SmootherOption::ScheduledRelaxedJacobi;

	m_rhs = lhs_matrix->createPressurePoissonRightHandSide(m_face_weight, m_packed_velocity.v[0],
		m_packed_velocity.v[1], m_packed_velocity.v[2], m_solid_velocity, in_dt);

	m_pressure = lhs_matrix->getZeroVectorGrid();
	m_pressure->setName("Pressure");
	CSim::TimerMan::timer("Step/SIMD/build").stop();
	m_substep_statistics.poisson_time_build += duration_type(CSim::TimerMan::timer("Step/SIMD/build").lasttime());
	m_substep_statistics.poisson_ndof = lhs_matrix->mNumDof;

	CSim::TimerMan::timer("Step/SIMD/solve").start();
	auto state = simd_solver.solveMultigridPCG(m_pressure, m_rhs);
	m_substep_statistics.poisson_pcg_iterations += simd_solver.mIterationTaken;

	if (state != simd_uaamg::PoissonSolver::SUCCESS) {
		printf("start MG solve\n");
		lhs_matrix->setGridToConstant(m_pressure, 0.f);
		simd_solver.solvePureMultigrid(m_pressure, m_rhs);
		m_substep_statistics.poisson_pcg_failed = true;
		m_substep_statistics.poisson_MG_iterations += simd_solver.mIterationTaken;
	}
	CSim::TimerMan::timer("Step/SIMD/solve").stop();
	m_substep_statistics.poisson_time_solve += duration_type(CSim::TimerMan::timer("Step/SIMD/solve").lasttime());

	//CSim::TimerMan::timer("Step/SIMD/grad").start();
	apply_pressure_gradient(in_dt, reflection);
	//CSim::TimerMan::timer("Step/SIMD/grad").stop();
	/*m_rhs = lhs_matrix->createPressurePoissonRightHandSide(m_face_weight, m_packed_velocity.v[0],
		m_packed_velocity.v[1], m_packed_velocity.v[2], m_solid_velocity, in_dt);*/
	//m_rhs = simd_solver.mMultigridHierarchy[0]->mDiagonal;
	m_rhs->setName("RHS");
}

void FLIP_Solver_OpenVDB::solve_viscosity(FLIP_Object_OpenVDB& in_out_object, float in_dt)
{
	using duration_type = std::chrono::duration<double>;
	float viscosity = in_out_object.m_viscosity;
	auto viscosity_grid = openvdb::FloatGrid::create(viscosity);
	CSim::TimerMan::timer("Step/viscosity/build_hierarchy").start();
	extrapolate_velocity(3);

	simd_uaamg::simd_viscosity3d viscosity_solver(viscosity_grid,
		m_liquid_sdf,
		m_solid_sdf,
		m_packed_velocity,
		m_solid_velocity, in_dt, /*density*/in_out_object.m_density);
	CSim::TimerMan::timer("Step/viscosity/build_hierarchy").stop();
	m_substep_statistics.viscosity_time_build += duration_type(CSim::TimerMan::timer("Step/viscosity/build_hierarchy").lasttime());

	auto result = viscosity_solver.m_matrix_levels[0]->get_zero_vec();

	//warm start
	for (int i = 0; i < 3; i++) {
		auto warm_start_setter = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) {
			auto original_vel_axr = m_packed_velocity.v[i]->getConstUnsafeAccessor();
			auto result_leaf = result.v[i]->tree().probeLeaf(leaf.origin());
			for (auto iter = leaf.beginValueOn(); iter; ++iter) {
				result_leaf->setValueOnly(iter.offset(), original_vel_axr.getValue(iter.getCoord()));
			}
		};
		viscosity_solver.m_matrix_levels[0]->m_dof_manager[i]->foreach(warm_start_setter);
	}
	static int counter = 0;
	result.setName("result_before_solve");

	//openvdb::io::File("dbgpre"+std::to_string(counter)+".vdb").write({result.v[0], result.v[1], result.v[2] });
	CSim::TimerMan::timer("Step/viscosity/solve").start();
	viscosity_solver.pcg_solve(result, m_solver_parameters.viscosity_solver_accuracy);
	CSim::TimerMan::timer("Step/viscosity/solve").stop();
	m_substep_statistics.viscosity_time_solve += duration_type(CSim::TimerMan::timer("Step/viscosity/solve").lasttime());
	m_substep_statistics.viscosity_ndof = viscosity_solver.m_matrix_levels[0]->m_ndof;
	m_substep_statistics.viscosity_iterations = viscosity_solver.m_iteration;

	//openvdb::io::File("dbgpost" + std::to_string(counter) + ".vdb").write({ result.v[0], result.v[1], result.v[2] });
	//result = m_packed_velocity.deepCopy();
	//viscosity_solver.m_matrix_levels[0]->solve_exact(result, viscosity_solver.m_rhs);
	//vdb_velocity_extrapolator::union_extrapolate(3, result.v[0], result.v[1], result.v[2], &(m_liquid_sdf->tree()));
	m_solved_viscosity_velocity = result.deepCopy();
	m_solved_viscosity_velocity.setName("m_solved_viscosity_velocity");

	for (int i = 0; i < 3; i++) {
		m_packed_velocity.v[i]->setTree(result.v[i]->treePtr());
	}
}

void FLIP_Solver_OpenVDB::update_solid_sdf_vel(openvdb::points::PointDataGrid::Ptr in_out_particles, const std::vector<sdf_vel_pair>& in_sdf_vels)
{
	ZoneScoped;
	using  namespace openvdb::tools::local_util;
	//solid is only required where particles present
	//derive it from particles, by default the space is outside of solid, so it's positive

	m_solid_sdf->setTree(std::make_shared<openvdb::FloatTree>(
		in_out_particles->tree(), 3.0f * m_dx, openvdb::TopologyCopy()));
	openvdb::tools::dilateActiveValues(m_solid_sdf->tree(), 4, openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);

	m_solid_sdf->setTransform(m_voxel_vertex_transform);
	m_solid_sdf->setName("solid_sdf");

	m_solid_velocity->setTree(std::make_shared<openvdb::Vec3fTree>(
		m_solid_sdf->tree(), openvdb::Vec3f{ 0.f }, openvdb::TopologyCopy()));
	m_solid_velocity->setTransform(m_voxel_center_transform);
	m_solid_velocity->setName("solid_velocity");

	std::vector<openvdb::FloatTree::LeafNodeType*> sdfleaves;
	sdfleaves.reserve(m_solid_sdf->tree().leafCount());
	m_solid_sdf->tree().getNodes(sdfleaves);

	std::vector<openvdb::Vec3fTree::LeafNodeType*> velleaves;
	velleaves.reserve(m_solid_velocity->tree().leafCount());
	m_solid_velocity->tree().getNodes(velleaves);
	std::vector<int> solid_leaf_modified;
	solid_leaf_modified.assign(sdfleaves.size(), 0);

	//active solids
	std::vector<sdf_vel_pair> on_sdf_vels;
	for (auto& pair : in_sdf_vels) {
		if (pair.m_activated) {
			on_sdf_vels.push_back(pair);
		}
	}

	auto set_solid_sdf_vel = [&](const tbb::blocked_range<size_t>& r) {
		std::vector<openvdb::tree::ValueAccessor<const openvdb::FloatTree>> solid_axrs;
		std::vector<openvdb::tree::ValueAccessor<const openvdb::Vec3fTree>> vel_axrs;

		for (size_t isolid = 0; isolid < on_sdf_vels.size(); isolid++) {
			solid_axrs.push_back(openvdb::tree::ValueAccessor<const openvdb::FloatTree>{ on_sdf_vels[isolid].m_sdf->tree() });
			vel_axrs.push_back(openvdb::tree::ValueAccessor<const openvdb::Vec3fTree>{ on_sdf_vels[isolid].m_vel->tree() });
		}

		//broadphase detect if a bulk can potentially have any solid inside
		auto touch_solid = [&](const openvdb::Vec3f& wpos, float delta = 0) {
			//check solid
			for (int isolid = 0; isolid < on_sdf_vels.size(); isolid++) {
				openvdb::Vec3f solidipos = on_sdf_vels[isolid].m_sdf->worldToIndex(wpos);
				if (solid_axrs[isolid].probeConstLeaf(openvdb::Coord(floorVec3(solidipos)))) {
					return true;
				}
				//detech deep inside solid space without bulks
				if (openvdb::tools::BoxSampler::sample(solid_axrs[isolid], solidipos) < delta) {
					return true;
				}
			}
			return false;
		};//end touch_solid

		float touch_threshold = m_dx;
		for (auto leafpos = r.begin(); leafpos != r.end(); leafpos++) {
			auto leaforigin = sdfleaves[leafpos]->origin();

			bool corner_solid = false;
			//anyone of its corners touch sink zone then it is worth processing
			for (int i = 0; i <= 8 && !corner_solid; i += 4) {
				for (int j = 0; j <= 8 && !corner_solid; j += 4) {
					for (int k = 0; k <= 8 && !corner_solid; k += 4) {
						auto voxelwpos = in_out_particles->indexToWorld(leaforigin.offsetBy(i, j, k) - openvdb::Vec3f(0.5));
						corner_solid |= touch_solid(voxelwpos, touch_threshold);
					}//k
				}//j
			}//i

			if (!corner_solid) {
				sdfleaves[leafpos]->setValuesOff();
				velleaves[leafpos]->setValuesOff();
				continue;
			}

			//mark this solid leaf as modified
			solid_leaf_modified[leafpos] = 1;
			//below this line, this bulk potentially touches a solid
			for (auto offset = 0; offset < sdfleaves[leafpos]->SIZE; ++offset) {
				auto voxelwpos = sdfleaves[leafpos]->offsetToGlobalCoord(offset);
				float temp_sdf = m_solid_sdf->background();
				openvdb::Vec3f sampled_vel{ 0 };
				float velocity_weight = 0.f;
				for (int isolid = 0; isolid < on_sdf_vels.size(); ++isolid) {
					auto solidipos = on_sdf_vels[isolid].m_sdf->worldToIndex(m_solid_sdf->indexToWorld(voxelwpos));
					auto sampled_sdf = openvdb::tools::BoxSampler::sample(solid_axrs[isolid], solidipos);
					if (sampled_sdf < temp_sdf) {
						temp_sdf = sampled_sdf;
						//sampled_vel = openvdb::tools::BoxSampler::sample(vel_axrs[isolid], solidipos);
					}
					if (sampled_sdf < m_dx) {
						float temp_weight = 1.0f / (std::abs(sampled_sdf) + 0.5f * m_dx);
						velocity_weight += temp_weight;
						sampled_vel += temp_weight * openvdb::tools::BoxSampler::sample(vel_axrs[isolid], solidipos);
					}
				}

				if (temp_sdf != m_solid_sdf->background()) {
					sdfleaves[leafpos]->setValueOn(offset, temp_sdf);
					if (velocity_weight != 0.f) {
						sampled_vel /= velocity_weight;
					}
					velleaves[leafpos]->setValueOn(offset, sampled_vel);
				}
				else {
					sdfleaves[leafpos]->setValueOff(offset);
					velleaves[leafpos]->setValueOff(offset);
				}
			}//end for all voxels in this sdf
		}//end for range leaves
	};//end set_solid_sdf lambda

	//CSim::TimerMan::timer("Step/update_solid_sdf/set").start();
	tbb::parallel_for(tbb::blocked_range<size_t>(0, sdfleaves.size()), set_solid_sdf_vel);
	//CSim::TimerMan::timer("Step/update_solid_sdf/set").stop();
	//CSim::TimerMan::timer("Step/update_solid_sdf/prune").start();

	for (size_t leafpos = 0; leafpos < solid_leaf_modified.size(); ++leafpos) {
		if (!solid_leaf_modified[leafpos]) {
			delete m_solid_sdf->tree().stealNode<openvdb::FloatTree::LeafNodeType>(sdfleaves[leafpos]->origin(), m_solid_sdf->background(), false);
			delete m_solid_velocity->tree().stealNode<openvdb::Vec3fTree::LeafNodeType>(sdfleaves[leafpos]->origin(), m_solid_velocity->background(), false);
		}
	}
}

FLIP_Object_OpenVDB::FLIP_Object_OpenVDB(float in_dx)
{
	openvdb::initialize();
	m_particles = openvdb::points::PointDataGrid::create();
	m_particles->setTransform(openvdb::math::Transform::createLinearTransform(in_dx));
	m_particles->setName("Particles");
	m_particle_radius = in_dx * 0.5f * 1.732f * 1.01f;
	printf("FLIP Object Ctor done\n");
}

FLIP_Object_OpenVDB::FLIP_Object_OpenVDB(const sdf_vel_pair& in_sdfvel, float in_dx)
{
	openvdb::initialize();
	m_particles = openvdb::points::PointDataGrid::create();
	m_particles->setTransform(openvdb::math::Transform::createLinearTransform(in_dx));
	m_particles->setName("Particles");
	m_particle_radius = in_dx * 0.5f * 1.732f * 1.01f;

	auto emitter_bbox = in_sdfvel.m_sdf->evalActiveVoxelBoundingBox();
	auto emittermin = in_sdfvel.m_sdf->indexToWorld(emitter_bbox.min());
	auto emittermax = in_sdfvel.m_sdf->indexToWorld(emitter_bbox.max());
	FLIP_Solver_OpenVDB::emit_liquid(m_particles, in_sdfvel, emittermin, emittermax);
	m_velocity.from_vec3(in_sdfvel.m_vel);
}

FLIP_Object_OpenVDB::FLIP_Object_OpenVDB(const Compressed_FLIP_Object_OpenVDB& in_compressed_object)
{
	m_viscosity = in_compressed_object.surface_layer_particles->
		getMetadata<openvdb::TypedMetadata<float>>("viscosity")->value();
	m_density = in_compressed_object.surface_layer_particles->
		getMetadata<openvdb::TypedMetadata<float>>("density")->value();
	m_particle_radius = in_compressed_object.surface_layer_particles->
		getMetadata<openvdb::TypedMetadata<float>>("particle_radius")->value();
	m_particles = in_compressed_object.surface_layer_particles->deepCopy();

	float dx = static_cast<float>(m_particles->voxelSize()[0]);
	auto interior_sdf = openvdb::createLevelSet<openvdb::FloatGrid>(m_particles->voxelSize()[0]);
	interior_sdf->setTree(std::make_shared<openvdb::FloatTree>(
		/*tree*/in_compressed_object.interior_mask->tree(),
		/*inactive value*/interior_sdf->background(),
		/*active value*/-dx, openvdb::TopologyCopy()));

	m_velocity.from_vec3(in_compressed_object.velocity);
	sdf_vel_pair source_pair{ interior_sdf, in_compressed_object.velocity };
	auto emitter_bbox = source_pair.m_sdf->evalActiveVoxelBoundingBox();
	auto emittermin = source_pair.m_sdf->indexToWorld(emitter_bbox.min());
	auto emittermax = source_pair.m_sdf->indexToWorld(emitter_bbox.max());
	FLIP_Solver_OpenVDB::emit_liquid(m_particles, source_pair, emittermin, emittermax);
}

Compressed_FLIP_Object_OpenVDB::Compressed_FLIP_Object_OpenVDB(const FLIP_Object_OpenVDB& in_object)
{
	surface_layer_particles = openvdb::points::PointDataGrid::create();
	auto interior_sdf = openvdb::FloatGrid::create();
	Compressed_FLIP_Object_OpenVDB::narrow_band_particles(surface_layer_particles, interior_sdf, in_object.m_particles, 5);
	surface_layer_particles->setName("Particles");
	surface_layer_particles->setTransform(in_object.m_particles->transform().copy());
	surface_layer_particles->insertMeta("viscosity", openvdb::TypedMetadata<float>(in_object.m_viscosity));
	surface_layer_particles->insertMeta("density", openvdb::TypedMetadata<float>(in_object.m_density));
	surface_layer_particles->insertMeta("particle_radius", openvdb::TypedMetadata<float>(in_object.m_particle_radius));

	interior_mask = openvdb::BoolGrid::create();
	interior_mask->topologyUnion(*interior_sdf);
	interior_mask->setTransform(surface_layer_particles->transformPtr());
	interior_mask->setName("interior_mask");

	velocity = openvdb::Vec3fGrid::create();
	in_object.m_velocity.to_vec3(velocity);
	velocity->setTransform(surface_layer_particles->transformPtr());
	velocity->setName("velocity");
}

void Compressed_FLIP_Object_OpenVDB::write(std::string filename) const
{
	openvdb::io::File(filename).write({ surface_layer_particles, velocity, interior_mask });
}
