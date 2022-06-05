#pragma once
#include "openvdb/openvdb.h"
#include "openvdb/tools/MeshToVolume.h"
#include "openvdb/tree/LeafManager.h"
#include "openvdb/math/Proximity.h"
#include <fstream>
#include <string>
//#include <string>

struct animation_utils {
	struct MeshDataAdapter {
		MeshDataAdapter(float in_voxel_size)
		{
			m_voxel_size = in_voxel_size;
			m_inv_voxel_size = 1.0f / m_voxel_size;
		};

		// Total number of polygons
		size_t polygonCount() const {
			return m_triangle_vtx_idx.size();
		};

		// Total number of points
		size_t pointCount() const {
			return m_vtx_pos.size();
		};

		// Vertex count for polygon n
		size_t vertexCount(size_t n) const { return 3; };

		// Return position pos in local grid index space for polygon n and vertex v
		void getIndexSpacePoint(size_t n, size_t v, openvdb::Vec3d& pos) const {
			auto	vtxs = m_triangle_vtx_idx[n];
			//turns original world space data into grid index space.
			auto vidx = vtxs[v];
			pos = m_vtx_pos[vtxs[v]] * m_inv_voxel_size;
		};

		std::vector<openvdb::Vec3f> m_vtx_pos;
		std::vector<openvdb::Vec3I> m_triangle_vtx_idx;
		float m_voxel_size;
		float m_inv_voxel_size;
	};


	static void readMeshFromObj(MeshDataAdapter & adapter, std::string filename) {
		std::fstream input_file;
		input_file.open(filename);
		adapter.m_vtx_pos.clear();
		adapter.m_triangle_vtx_idx.clear();

		std::string str;
		while (std::getline(input_file, str)) {
			// process string ...
			std::stringstream ss(str);
			char first_letter;
			ss >> first_letter;
			if (first_letter == 'v') {
				float v0, v1, v2;
				ss >> v0 >> v1 >> v2;
				adapter.m_vtx_pos.push_back(openvdb::Vec3f(v0, v1, v2));
				continue;
			}
			if (first_letter == 'f') {
				int i0, i1, i2;
				ss >> i0 >> i1 >> i2;
				adapter.m_triangle_vtx_idx.push_back(openvdb::Vec3I(i0 - 1, i1 - 1, i2 - 1));
				continue;
			}
		}
		input_file.close();
	}

	static void interpolationMesh(MeshDataAdapter& out_adapter, const MeshDataAdapter& adapter0, const MeshDataAdapter& adapter1, float alpha) {
		out_adapter = adapter0;
		for (int i = 0; i < adapter0.m_vtx_pos.size(); ++i) {
			out_adapter.m_vtx_pos[i] = (1 - alpha) * adapter0.m_vtx_pos[i] + alpha * adapter1.m_vtx_pos[i];
		}
	}

	static void calculate_velocity(std::vector<openvdb::Vec3f>& out_vtx_vel, const MeshDataAdapter& adapter0, const MeshDataAdapter& adapter1, float dt) {
		out_vtx_vel = adapter0.m_vtx_pos;
		for (int i = 0; i < out_vtx_vel.size(); i++) {
			out_vtx_vel[i] = (adapter1.m_vtx_pos[i] - adapter0.m_vtx_pos[i]) / dt;
		}
	}

	static void get_sdf_and_velocity_field(
		openvdb::FloatGrid::Ptr out_sdf,
		openvdb::Vec3fGrid::Ptr out_vel,
		const MeshDataAdapter& adapter0,
		const MeshDataAdapter& adapter1, float alpha, float frame_dt) {
		MeshDataAdapter intermediate_mesh{ adapter0 };
		interpolationMesh(intermediate_mesh, adapter0, adapter1, alpha);
		std::vector<openvdb::Vec3f> vtx_velocity;
		calculate_velocity(vtx_velocity, adapter0, adapter1, frame_dt);

		//voxel size is the edge length
		auto voxel_size = adapter0.m_voxel_size;

		openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform(voxel_size);
		auto m_idxgrid = openvdb::Int32Grid::create(openvdb::util::INVALID_IDX);
		//printf("runm2v\n");
		auto liquid_SDF = openvdb::tools::meshToVolume<openvdb::FloatGrid, MeshDataAdapter>(intermediate_mesh, *transform, 3.0f, 3.0f, 0, m_idxgrid.get());
		liquid_SDF->setName("collision");
		m_idxgrid->setName("idxgrid");
		out_sdf->setTransform(liquid_SDF->transformPtr());
		out_sdf->setTree(liquid_SDF->treePtr());
		out_sdf->setName("collision");
		//calculate the average velocity of all vertices

		/*for (size_t i = 0; i < m_vertex_velocities.size(); ++i) {
			total_vel += m_vertex_velocities[i];
		}
		if (m_vertex_velocities.size() != 0) {
			total_vel /= m_vertex_velocities.size();
		}*/


		//given the closest index on the index grid
		//calculate the velocity
		auto m_nearest_velocity = openvdb::Vec3fGrid::create();
		m_nearest_velocity->setName("collision_velocity");
		m_nearest_velocity->setTransform(liquid_SDF->transformPtr());
		m_nearest_velocity->setTree(std::make_shared<openvdb::Vec3fTree>(
			m_idxgrid->tree(), openvdb::Vec3f{ 0,0,0 }, openvdb::TopologyCopy()
			));

		auto sample_vel_op = [&](openvdb::Int32Tree::LeafNodeType& leaf, openvdb::Index leafpos) {
			auto vel_leaf = m_nearest_velocity->tree().probeLeaf(leaf.origin());

			openvdb::Vec3d vtx[3];
			openvdb::Vec3d uvw;
			openvdb::Vec3d p;
			openvdb::Vec3f vel[3];
			for (auto iter = leaf.cbeginValueOn(); iter; ++iter) {
				if (iter.getValue() < 0) {
					continue;
				}
				//get its world coordinates
				auto voxel_wpos = m_idxgrid->indexToWorld(iter.getCoord());
				auto primidx = iter.getValue();
				const auto& tris = intermediate_mesh.m_triangle_vtx_idx[primidx];

				for (int vid = 0; vid < 3; vid++) {
					vtx[vid] = intermediate_mesh.m_vtx_pos[tris[vid]];
					vel[vid] = vtx_velocity[tris[vid]];
				}

				//get the barycentric coordinates
				openvdb::math::closestPointOnTriangleToPoint(vtx[0], vtx[1], vtx[2], voxel_wpos, uvw);
				openvdb::Vec3f mixed_v{ 0,0,0 };
				for (int vid = 0; vid < 3; vid++) {
					mixed_v += vel[vid] * float(uvw[vid]);
				}
				vel_leaf->setValueOn(iter.offset(), mixed_v);
			}
		};

		auto idxleafman = openvdb::tree::LeafManager<openvdb::Int32Tree>(m_idxgrid->tree());
		idxleafman.foreach(sample_vel_op);
		out_vel->setTransform(m_nearest_velocity->transformPtr());
		out_vel->setName("solid_velocity");
		out_vel->setTree(m_nearest_velocity->treePtr());
	}
};//end namespace animation_utils