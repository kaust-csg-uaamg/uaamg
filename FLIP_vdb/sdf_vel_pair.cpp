#include "sdf_vel_pair.h"

sdf_vel_pair::sdf_vel_pair(openvdb::FloatGrid::Ptr in_sdf, openvdb::Vec3fGrid::Ptr in_vel)
{
	init();
	m_sdf = in_sdf;
	m_vel = in_vel;
}

sdf_vel_pair::sdf_vel_pair(openvdb::FloatGrid::Ptr in_sdf, openvdb::Vec3f in_vel)
{
	init();
	m_sdf = in_sdf;
	m_vel = openvdb::Vec3fGrid::create(in_vel);
}

void sdf_vel_pair::setsdf(openvdb::FloatGrid::Ptr in_sdf)
{
	m_sdf = in_sdf;
}

void sdf_vel_pair::setvel(openvdb::Vec3fGrid::Ptr in_vel)
{
	m_vel = in_vel;
}

void sdf_vel_pair::setvel(openvdb::Vec3f in_vel)
{
	m_vel = openvdb::Vec3fGrid::create(in_vel);
}

void sdf_vel_pair::init()
{
	m_sdf = openvdb::FloatGrid::create(1.0f);
	m_vel = openvdb::Vec3fGrid::create(openvdb::Vec3f{ 0.f });
	m_activated = true;
}
