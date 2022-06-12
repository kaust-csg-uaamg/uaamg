#ifdef WIN32
#if !defined NOMINMAX
#define NOMINMAX
#endif
#include <direct.h>
#endif
#include "tbb/tbbmalloc_proxy.h"
#include "FLIP_vdb.h"
#include "YamlSingleton.h"
#include "openvdb/tools/MeshToVolume.h"
#include "openvdb/tools/LevelSetSphere.h"
#include "animation_utils.h"
#include "vdb_velocity_extrapolator.h"

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cassert>
#include <thread>
#include <filesystem>
#include <ios>

#define add_time(name) in_out_node[#name] = statobj.name.count()
#define add_val(name) in_out_node[#name] = statobj.name;

void write_substep_stat_data(YAML::Node& in_out_node, FLIP_Solver_OpenVDB& in_FLIP_solver) {
    auto& statobj = in_FLIP_solver.m_substep_statistics;
    add_time(advection_time);
    add_time(p2g_time);
    add_time(emission_time);
    add_time(sink_time);
    add_time(update_solid_sdf_vel_time);
    add_time(extrapolate_time);
    add_time(substep_total_time);
    add_val(particle_count);

    add_time(poisson_time_total);
    add_time(poisson_time_build);
    add_time(poisson_time_solve);
    add_val(poisson_pcg_iterations);
    add_val(poisson_MG_iterations);
    add_val(poisson_pcg_failed);
    add_val(poisson_ndof);

    add_time(viscosity_time_total);
    add_time(viscosity_time_build);
    add_time(viscosity_time_solve);
    add_val(viscosity_iterations);
    add_val(viscosity_ndof);
}


int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);

    // load yaml files
    if (argc != 2) {
        printf("Usage: FLIP_simulator configuration_file_name.yaml\n");
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

    std::string output_prefix = YamlSingleton::get()["output_prefix"].as<std::string>();
    std::filesystem::path fluid_output_directory(output_prefix);
    std::filesystem::create_directory(fluid_output_directory);
    std::ofstream fout((fluid_output_directory/"config.yaml").string());
    fout << YamlSingleton::get();
    fout.close();
    fluid_output_directory /= "vdb";
    std::filesystem::create_directory(fluid_output_directory);

    std::filesystem::path stats_output_directory(output_prefix);
    stats_output_directory /= "stats";
    std::filesystem::create_directory(stats_output_directory);
    openvdb::initialize();

    float FLIPdx = YamlSingleton::get()["FLIPdx"].as<float>();
    float minx = YamlSingleton::get()["FLIPpool-minx"].as<float>();
    float miny = YamlSingleton::get()["FLIPpool-miny"].as<float>();
    float minz = YamlSingleton::get()["FLIPpool-minz"].as<float>();
    float maxx = YamlSingleton::get()["FLIPpool-maxx"].as<float>();
    float maxy = YamlSingleton::get()["FLIPpool-maxy"].as<float>();
    float maxz = YamlSingleton::get()["FLIPpool-maxz"].as<float>();
    float init_vel_x = YamlSingleton::get()["init_vel_x"].as<float>();
    float init_vel_y = YamlSingleton::get()["init_vel_y"].as<float>();
    float init_vel_z = YamlSingleton::get()["init_vel_z"].as<float>();

    auto flowv = openvdb::Vec3f{ init_vel_x, init_vel_y, init_vel_z };

    std::unique_ptr<FLIP_Object_OpenVDB> flip_object;
    std::unique_ptr<FLIP_Solver_OpenVDB> vdbflip;
    printf("initializing FLIP object\n");
    //Initialize the FLIP object

    if (YamlSingleton::get()["source_vdb"]) {
       
        openvdb::io::File input_source(YamlSingleton::get()["source_vdb"].as<std::string>());
        try { 
            std::cout << "load liquid source vdb file: " << input_source.filename() << std::endl;
            input_source.open(); 
        }
        catch (openvdb::IoError err) {
            std::cout << "cannot open file: " << YamlSingleton::get()["source_vdb"].as<std::string>() << '\n';
            return 1;
        } 

        openvdb::GridBase::Ptr sourceGrid;
        sdf_vel_pair source_sdf_vel;
        for (openvdb::io::File::NameIterator nameIter = input_source.beginName();
            nameIter != input_source.endName(); ++nameIter)
        {
            if (nameIter.gridName() == "initial_liquid_sdf") {
                sourceGrid = input_source.readGrid(nameIter.gridName());
                source_sdf_vel =  sdf_vel_pair(openvdb::DynamicPtrCast<openvdb::FloatGrid>(sourceGrid), openvdb::Vec3fGrid::create(flowv));
            }
            if (nameIter.gridName() == "vel") {
                sourceGrid = input_source.readGrid(nameIter.gridName());
                //this is the corresponding velocity of the previous source
                source_sdf_vel.m_vel = openvdb::DynamicPtrCast<openvdb::Vec3fGrid>(sourceGrid);
            }
        }
        printf("init FLIP Obj from vdb source\n");
        flip_object = std::make_unique<FLIP_Object_OpenVDB>(source_sdf_vel, FLIPdx);
        input_source.close();
    }

    if (!flip_object) {
        printf("init FLIP Obj from dx\n");
        flip_object = std::make_unique<FLIP_Object_OpenVDB>(FLIPdx);
    }
    
    printf("initializing FLIP object done\n");
    FLIP_Solver_OpenVDB::SolverParameters solver_param;
    solver_param.cfl = YamlSingleton::get()["FLIPCFL"].as<float>();
    solver_param.domain_min = openvdb::Vec3f(minx, miny, minz);
    solver_param.domain_max = openvdb::Vec3f(maxx, maxy, maxz);

    solver_param.pressure_solver_accuracy = YamlSingleton::get()["pressure_acc"].as<float>();
    if (YamlSingleton::get()["enable_viscosity"] && YamlSingleton::get()["viscosity_coef"]) {
        if (YamlSingleton::get()["enable_viscosity"].as<bool>()) {
            solver_param.enable_viscosity = FLIP_Solver_OpenVDB::ViscosityOption::ON;
            flip_object->set_viscosity(YamlSingleton::get()["viscosity_coef"].as<float>());
            solver_param.viscosity_solver_accuracy = YamlSingleton::get()["viscosity_acc"].as<float>();
        }
    }

    vdbflip = std::make_unique<FLIP_Solver_OpenVDB>(*flip_object, solver_param);
    printf("initializing FLIP Solver done\n");
    if (YamlSingleton::get()["source_vdb"]) {
        openvdb::io::File input_source(YamlSingleton::get()["source_vdb"].as<std::string>());
        input_source.open();
        openvdb::GridBase::Ptr sourceGrid;
        for (openvdb::io::File::NameIterator nameIter = input_source.beginName();
            nameIter != input_source.endName(); ++nameIter)
        {
            // Read in only the grid we are interested in.
            if (nameIter.gridName() == "source") {
                sourceGrid = input_source.readGrid(nameIter.gridName());
                auto insource_sdfvel = sdf_vel_pair(openvdb::DynamicPtrCast<openvdb::FloatGrid>(sourceGrid), openvdb::Vec3fGrid::create(flowv));
                vdbflip->m_source_sdfvel.push_back(insource_sdfvel);
            }
            if (nameIter.gridName() == "vel") {
                sourceGrid = input_source.readGrid(nameIter.gridName());
                //this is the corresponding velocity of the previous source
                for (auto iter = vdbflip->m_source_sdfvel.rbegin(); iter != vdbflip->m_source_sdfvel.rend(); ++iter) {
                    if (iter->m_sdf->getName() == "source") {
                        iter->m_vel = openvdb::DynamicPtrCast<openvdb::Vec3fGrid>(sourceGrid);
                        break;
                    }
                }
            }
        }
        input_source.close();
    }
    printf("Add source done\n");
    

    //FLIP solids
    if (YamlSingleton::get()["solids_vdb"]) {

        openvdb::io::File input_solid(YamlSingleton::get()["solids_vdb"].as<std::string>());
        input_solid.open();
        openvdb::GridBase::Ptr solidGrid;
        for (openvdb::io::File::NameIterator nameIter = input_solid.beginName();
            nameIter != input_solid.endName(); ++nameIter)
        {
            // Read in only the grid we are interested in.
            if (nameIter.gridName() == "collision") {

                solidGrid = input_solid.readGrid(nameIter.gridName());
                auto collision_sdfvel = sdf_vel_pair(openvdb::DynamicPtrCast<openvdb::FloatGrid>(solidGrid), openvdb::Vec3fGrid::create(openvdb::Vec3f(0)));
                vdbflip->m_collision_sdfvel.push_back(collision_sdfvel);
                vdbflip->m_solid_sdfvel.push_back(collision_sdfvel);
            }

            if (nameIter.gridName() == "solid") {
                solidGrid = input_solid.readGrid(nameIter.gridName());
                auto solid_sdfvel = sdf_vel_pair(openvdb::DynamicPtrCast<openvdb::FloatGrid>(solidGrid), openvdb::Vec3fGrid::create(flowv));
                vdbflip->m_solid_sdfvel.push_back(solid_sdfvel);
            }
            std::cout << "solid read done\n";
        }
        input_solid.close();
    }

    printf("Add solids done\n");
    if (YamlSingleton::get()["FLIP_domain_collision"].as<bool>() || YamlSingleton::get()["FLIP_domain_solid"].as<bool>()) {
        std::cout << "add domain to boundary\n";
        auto domain_sdfvel = sdf_vel_pair(vdbflip->get_domain_solid_sdf(), openvdb::Vec3fGrid::create(flowv));
        vdbflip->m_solid_sdfvel.push_back(domain_sdfvel);
        if (YamlSingleton::get()["FLIP_domain_collision"].as<bool>()) {
            vdbflip->m_collision_sdfvel.push_back(domain_sdfvel);
        }
    }

    std::vector<sdf_vel_pair> static_collision_sdfvel = vdbflip->m_collision_sdfvel;
    std::vector<sdf_vel_pair> static_solid_sdfvel = vdbflip->m_solid_sdfvel;

    printf("Add collision done\n");

    //FLIP sink
    if (YamlSingleton::get()["sink_vdb"]) {
        std::cout <<"sink: "<< YamlSingleton::get()["sink_vdb"].as<std::string>();
        openvdb::io::File input_sink(YamlSingleton::get()["sink_vdb"].as<std::string>());
        input_sink.open();
        openvdb::GridBase::Ptr sinkGrid;
        for (openvdb::io::File::NameIterator nameIter = input_sink.beginName();
            nameIter != input_sink.endName(); ++nameIter)
        {
            // Read in only the grid we are interested in.
            if (nameIter.gridName() == "sink") {
                sinkGrid = input_sink.readGrid(nameIter.gridName());
                vdbflip->m_sink_sdfs.push_back(openvdb::DynamicPtrCast<openvdb::FloatGrid>(sinkGrid));
            }
        }
        input_sink.close();
    }
    printf("Add sink done\n");

    const float dt = YamlSingleton::get()["time_step"].as<float>();
    const float max_time = YamlSingleton::get()["simulation_time"].as<float>();


    //initial dynamic objects
    std::unique_ptr<animation_utils::MeshDataAdapter> pmesh0, pmesh1;
    pmesh0 = std::make_unique<animation_utils::MeshDataAdapter>(FLIPdx);
    pmesh1 = std::make_unique<animation_utils::MeshDataAdapter>(FLIPdx);

    //simulation loop
    int framenumber = 0;
    for (framenumber = 0; framenumber * dt < max_time; framenumber++) {
        std::vector<std::thread> iothreads;

        //Compressed_FLIP_Object_OpenVDB compressed_obj(*flip_object);
        auto surface_particles = openvdb::points::PointDataGrid::create();
        auto interior_sdf = openvdb::FloatGrid::create();
        Compressed_FLIP_Object_OpenVDB::narrow_band_particles(surface_particles, interior_sdf, flip_object->get_particles(), /*nlayer*/2);
        auto vel = vdbflip->m_packed_velocity.deepCopy();
        auto liquidsdf = vdbflip->m_liquid_sdf->deepCopy();
        //auto solidsdf = vdbflip->m_solid_sdf->deepCopy();
        //auto solid_vel = vdbflip->m_solid_velocity->deepCopy();
		iothreads.push_back(std::thread{ [=]() {
			std::stringstream output_dir_ss;
			output_dir_ss << output_prefix << "/vdb/flip_particle"<< framenumber <<".vdb";
			std::cout << output_dir_ss.str() << std::endl;
            openvdb::io::File(output_dir_ss.str()).write({ surface_particles, interior_sdf});

            if (YamlSingleton::get()["output_sdfvel"] && YamlSingleton::get()["output_sdfvel"].as<bool>()) {
                std::stringstream output_sdfvel_ss;
                output_sdfvel_ss << output_prefix << "/vdb/sdfvel" << framenumber << ".vdb";
                openvdb::io::File(output_sdfvel_ss.str()).write({ liquidsdf, vel.v[0], vel.v[1], vel.v[2] });
            }
            } });

        printf("step\n");
        //run FLIP simulation

        //process dynamic collision
        if (YamlSingleton::get()["dynamic_obj_enabled"] && YamlSingleton::get()["dynamic_obj_enabled"].as<bool>()) {
            std::string dynamic_obj_folder = YamlSingleton::get()["dynamic_obj_folder"].as<std::string>();
            std::filesystem::path dynamic_obj_folder_path(dynamic_obj_folder);
            std::string dynamic_obj_prefix = YamlSingleton::get()["dynamic_obj_prefix"].as<std::string>();
            auto mesh0path = dynamic_obj_folder_path / (dynamic_obj_prefix + std::to_string(framenumber) + ".obj");
            auto mesh1path = dynamic_obj_folder_path / (dynamic_obj_prefix + std::to_string(framenumber + 1) + ".obj");

            if (!std::filesystem::exists(mesh0path)) {
                std::cout << mesh0path << "does not exist, terminating\n";
                exit(-1);
            }
            if (!std::filesystem::exists(mesh1path)) {
                std::cout << mesh1path << "does not exist, terminating\n";
                exit(-1);
            }

            if (framenumber == 0) {
                //read both obj
                animation_utils::readMeshFromObj(*pmesh0, mesh0path.string());
                animation_utils::readMeshFromObj(*pmesh1, mesh1path.string());
            }
            else {
                pmesh0.swap(pmesh1);
                //read new mesh
                animation_utils::readMeshFromObj(*pmesh1, mesh1path.string());
            }
            
        }

        const float target_time = dt;
        float stepped = 0.f;
        int sub_step_count = 0;

        int FLIP_min_step_num = YamlSingleton::get()["FLIP_min_step_num"].as<int>();
        int FLIP_max_step_num = YamlSingleton::get()["FLIP_max_step_num"].as<int>();
        float min_allowed_dt = dt / FLIP_max_step_num;
        float max_allowed_dt = dt / FLIP_min_step_num;

        YAML::Node frame_stat;

        while (stepped < target_time && sub_step_count < FLIP_max_step_num) {
            auto cfl_dt =  vdbflip->param().cfl * vdbflip->particle_cfl(flip_object->get_particles());
            if (vdbflip->param().simulation_method == FLIP_Solver_OpenVDB::SimulationMethod::ADVECTION_REFLECITON) {
                cfl_dt *= 2.0f;
            }

            float substep_dt = cfl_dt;
            substep_dt = std::max(substep_dt, min_allowed_dt);
            substep_dt = std::min(substep_dt, max_allowed_dt);
            substep_dt *= 1.01f;

            float reaching_dt = stepped + substep_dt;

            if (stepped + substep_dt > target_time) {
                reaching_dt = target_time;
            }

            substep_dt = reaching_dt - stepped;
            stepped = reaching_dt;
            //process static collision
            vdbflip->m_collision_sdfvel = static_collision_sdfvel;
            vdbflip->m_solid_sdfvel = static_solid_sdfvel;

            //process dynamic collision
            if (YamlSingleton::get()["dynamic_obj_enabled"] && YamlSingleton::get()["dynamic_obj_enabled"].as<bool>()) {
                float alpha = stepped / target_time;
                sdf_vel_pair dynamic_sdfvel;
                animation_utils::get_sdf_and_velocity_field(dynamic_sdfvel.m_sdf, dynamic_sdfvel.m_vel, *pmesh0, *pmesh1, alpha, dt);
                vdbflip->m_collision_sdfvel.push_back(dynamic_sdfvel);
                vdbflip->m_solid_sdfvel.push_back(dynamic_sdfvel);
            }

            //debug output
			/*{
				Compressed_FLIP_Object_OpenVDB dbgcompressed_obj(*flip_object);
                auto dbgvel = vdbflip->m_packed_velocity.deepCopy();
                auto dbgvel_update = vdbflip->m_packed_velocity_update.deepCopy();
                auto dbgpressure = vdbflip->m_pressure->deepCopy();
                auto dbgrhs = vdbflip->m_rhs->deepCopy();
				auto dbgliquidsdf = vdbflip->m_liquid_sdf->deepCopy();
				auto dbgsolidsdf = vdbflip->m_solid_sdf->deepCopy();
				auto dbgsolid_vel = vdbflip->m_solid_velocity->deepCopy();
                auto collision_sdf = vdbflip->m_collision_sdfvel.back().m_sdf;
                auto dbg_face_weight = vdbflip->m_face_weight->deepCopy();
				std::stringstream dbgoutput_dir_ss;
				dbgoutput_dir_ss << output_prefix << "/vdb/dbgframe"<< framenumber<< "sub" << sub_step_count << ".vdb";
				std::cout << dbgoutput_dir_ss.str() << std::endl;
				openvdb::io::File(dbgoutput_dir_ss.str()).write({ 
                    dbgsolid_vel, dbgsolidsdf, dbgliquidsdf, 
                    dbgcompressed_obj.surface_layer_particles, 
                    dbgvel.v[0], dbgvel.v[1], dbgvel.v[2], collision_sdf,
                    dbgpressure, dbgrhs, dbgvel_update.v[0], dbgvel_update.v[1], dbgvel_update.v[2],
                    dbg_face_weight});

			}*/
            printf("substep:%d, toward %d/100 \n", sub_step_count, (int)std::floor(stepped / target_time * 100.0f));
            vdbflip->substep(*flip_object, substep_dt);
            sub_step_count++;

            //record substep stats
            YAML::Node substep_stat_node;
            write_substep_stat_data(substep_stat_node, *vdbflip);
            substep_stat_node["simulation_reaching_dt"] = reaching_dt;
            frame_stat["substep_stat"].push_back(substep_stat_node);
        }

        //vdbflip->solve(*flip_object, dt);
        {
            std::filesystem::path statfile_path(stats_output_directory);
            statfile_path /= "stats" + std::to_string(framenumber) + ".yaml";
            std::ofstream fout(statfile_path.string());
            fout << frame_stat;
            fout.close();
        }

        for (auto& t : iothreads) {
            t.join();
        }
    }

    std::stringstream output_dir_ss;
    output_dir_ss << output_prefix << "/vdb/flip_particle" << framenumber << ".vdb";
    std::cout << output_dir_ss.str() << std::endl;
    openvdb::io::File(output_dir_ss.str()).write({ flip_object->get_particles() });
	return 0;
}