#ifndef _LINE_DETECTION_H_
#define _LINE_DETECTION_H_
#pragma once

#include "CommonFunctions.h"

struct PLANE
{
	double scale;
	std::vector<std::vector<std::vector<cv::Point3d> > > lines3d;

	PLANE &operator =(const PLANE &info)
	{
		this->scale    = info.scale;
		this->lines3d     = info.lines3d;
		return *this;
	}
};

class LineDetection3D 
{
public:
	LineDetection3D();
	~LineDetection3D();

	void run( PointCloud<double> &data, int k, std::vector<PLANE> &planes, std::vector<std::vector<cv::Point3d> > &lines, std::vector<double> &ts );

	void pointCloudSegmentation( std::vector<std::vector<int> > &regions );

	void planeBased3DLineDetection( std::vector<std::vector<int> > &regions, std::vector<PLANE> &planes );

	void postProcessing( std::vector<PLANE> &planes, std::vector<std::vector<cv::Point3d> > &lines );

	// 
	void regionGrow( double thAngle, std::vector<std::vector<int> > &regions );

	void regionMerging( double thAngle, std::vector<std::vector<int> > &regions );

	bool maskFromPoint( std::vector<cv::Point2d> &pts2d, double radius, double &xmin, double &ymin, double &xmax, double &ymax, int &margin, cv::Mat &mask );

	void lineFromMask( cv::Mat &mask, int thLineLengthPixel, std::vector<std::vector<std::vector<cv::Point2d> > > &lines );

	void outliersRemoval( std::vector<PLANE> &planes );

	void lineMerging( std::vector<PLANE> &planes, std::vector<std::vector<cv::Point3d> > &lines );

public:
	int k;
	int pointNum;
	double scale, magnitd;
	std::vector<PCAInfo> pcaInfos;
	PointCloud<double> pointData;
};

#endif //_LINE_DETECTION_H_
