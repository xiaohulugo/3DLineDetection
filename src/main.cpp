#include <stdio.h>
#include <fstream>

#include "LineDetection3D.h"
#include "nanoflann.hpp"
#include "utils.h"
#include "Timer.h"

using namespace cv;
using namespace std;
using namespace nanoflann;

void readDataFromFile( std::string filepath, PointCloud<double> &cloud )
{
	cloud.pts.reserve(10000000);
	cout<<"Reading data ..."<<endl;

	// 1. read in point data
	std::ifstream ptReader( filepath );
	std::vector<cv::Point3d> lidarPoints;
	double x = 0, y = 0, z = 0, color = 0;
	double nx, ny, nz;
	int a = 0, b = 0, c = 0; 
	int labelIdx = 0;
	int count = 0;
	int countTotal = 0;
	if( ptReader.is_open() )
	{
		while ( !ptReader.eof() ) 
		{
			//ptReader >> x >> y >> z >> a >> b >> c >> labelIdx;
			//ptReader >> x >> y >> z >> a >> b >> c >> color;
			//ptReader >> x >> y >> z >> color >> a >> b >> c;
			//ptReader >> x >> y >> z >> a >> b >> c ;
			ptReader >> x >> y >> z;
			//ptReader >> x >> y >> z >> color;
			//ptReader >> x >> y >> z >> nx >> ny >> nz;

			cloud.pts.push_back(PointCloud<double>::PtData(x,y,z));

		}
		ptReader.close();
	}

	std::cout << "Total num of points: " << cloud.pts.size() << "\n";
}

void writeOutPlanes( string filePath, std::vector<PLANE> &planes, double scale )
{
	// write out bounding polygon result
	string fileEdgePoints = filePath + "planes.txt";
	FILE *fp2 = fopen( fileEdgePoints.c_str(), "w");
	for (int p=0; p<planes.size(); ++p)
	{
		int R = rand()%255;
		int G = rand()%255;
		int B = rand()%255;

		for (int i=0; i<planes[p].lines3d.size(); ++i)
		{
			for (int j=0; j<planes[p].lines3d[i].size(); ++j)
			{
				cv::Point3d dev = planes[p].lines3d[i][j][1] - planes[p].lines3d[i][j][0];
				double L = sqrt(dev.x*dev.x + dev.y*dev.y + dev.z*dev.z);
				int k = L/(scale/10);

				double x = planes[p].lines3d[i][j][0].x, y = planes[p].lines3d[i][j][0].y, z = planes[p].lines3d[i][j][0].z;
				double dx = dev.x/k, dy = dev.y/k, dz = dev.z/k;
				for ( int j=0; j<k; ++j)
				{
					x += dx;
					y += dy;
					z += dz;

					fprintf( fp2, "%.6lf   %.6lf   %.6lf    ", x, y, z );
					fprintf( fp2, "%d   %d   %d   %d\n", R, G, B, p );
				}
			}
		}
	}
	fclose( fp2 );
}

void writeOutLines( string filePath, std::vector<std::vector<cv::Point3d> > &lines, double scale )
{
	// write out bounding polygon result
	string fileEdgePoints = filePath + "lines.txt";
	FILE *fp2 = fopen( fileEdgePoints.c_str(), "w");
	for (int p=0; p<lines.size(); ++p)
	{
		int R = rand()%255;
		int G = rand()%255;
		int B = rand()%255;

		cv::Point3d dev = lines[p][1] - lines[p][0];
		double L = sqrt(dev.x*dev.x + dev.y*dev.y + dev.z*dev.z);
		int k = L/(scale/10);

		double x = lines[p][0].x, y = lines[p][0].y, z = lines[p][0].z;
		double dx = dev.x/k, dy = dev.y/k, dz = dev.z/k;
		for ( int j=0; j<k; ++j)
		{
			x += dx;
			y += dy;
			z += dz;

			fprintf( fp2, "%.6lf   %.6lf   %.6lf    ", x, y, z );
			fprintf( fp2, "%d   %d   %d   %d\n", R, G, B, p );
		}
	}
	fclose( fp2 );
}


void main() 
{
	string fileData = "D://Facade//data.txt";
	string fileOut  = "D://Facade//data";

	// read in data
	PointCloud<double> pointData; 
	readDataFromFile( fileData, pointData );

	int k = 20;
	LineDetection3D detector;
	std::vector<PLANE> planes;
	std::vector<std::vector<cv::Point3d> > lines;
	std::vector<double> ts;
	detector.run( pointData, k, planes, lines, ts );
	cout<<"lines number: "<<lines.size()<<endl;
	cout<<"planes number: "<<planes.size()<<endl;
	
	writeOutPlanes( fileOut, planes, detector.scale );
	writeOutLines( fileOut, lines, detector.scale );
}