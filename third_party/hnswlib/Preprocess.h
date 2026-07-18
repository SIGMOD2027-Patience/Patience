#pragma once

#include <fstream>
#include <queue>
#include <iostream>

struct knnBen
{
	unsigned N;
	unsigned num;
	int** indice;
	float** dist;
};
class Preprocess
{
public:
	float** data;
	float** queries;
	int n;
	int nq;
	int dim;
	float** Dists;
	knnBen benchmark;
	std::string data_file;
	std::string ben_file;
public:
	Preprocess(float** data_, float** queries, int n_, int nq_, int dim_, const std::string& ben_file_);
	void load_data(const std::string& path);
	void ben_make();
	void ben_save();
	void ben_load();
	void ben_create();
	~Preprocess();
};
