#pragma once
#include "yaml-cpp/yaml.h"
struct YamlSingleton {
	static YAML::Node& get() {
		static YAML::Node root;
		return root;
	};
	static void loadfile(const char* filename) {
		get() = YAML::LoadFile(filename);
	};
};