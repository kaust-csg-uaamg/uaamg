#pragma once
#include <openvdb/openvdb.h>
/*
signed distance field source manager
manages the existance of a series of sdf
and their beginning and end time, initial velocity

It parses a file containing lines descripting
the source obj file path, source begin and end time, source initial velocity
*/

class sdf_vel_pair {
public:
	sdf_vel_pair() { init(); };
	sdf_vel_pair(openvdb::FloatGrid::Ptr in_sdf, openvdb::Vec3fGrid::Ptr in_vel);
	sdf_vel_pair(openvdb::FloatGrid::Ptr in_sdf, openvdb::Vec3f in_vel);
	
	sdf_vel_pair(const sdf_vel_pair& other) {
		m_activated = other.m_activated;
		m_sdf = other.m_sdf;
		m_vel = other.m_vel;
	}

	void setsdf(openvdb::FloatGrid::Ptr in_sdf);
	void setvel(openvdb::Vec3fGrid::Ptr in_vel);
	void setvel(openvdb::Vec3f in_vel);

	void init();
	void activate() { m_activated = true; }
	void deactivate() { m_activated = false; }

	openvdb::FloatGrid::Ptr m_sdf;
	openvdb::Vec3fGrid::Ptr m_vel;
	bool m_activated;
};