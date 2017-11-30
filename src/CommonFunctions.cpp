#include "CommonFunctions.h"
  
using namespace cv;
using namespace std;

/************************************************************************/
/*                           Line Functions                             */
/************************************************************************/

void LineFunctions::lineFitting( int rows, int cols, std::vector<cv::Point> &contour, double thMinimalLineLength, std::vector<std::vector<cv::Point2d> > &lines )
{
	// get straight strings from the contour
	double minDeviation = 6.0;
	std::vector<std::vector<cv::Point> > straightString;
	subDivision(straightString, contour, 0, contour.size()-1, minDeviation, int(thMinimalLineLength));
	if ( !straightString.size() )
	{
		return;
	}
	for ( int i=0; i<straightString.size(); ++i )
	{
		if ( straightString[i].size() < thMinimalLineLength )
		{
			continue;
		}

		std::vector<double> parameters( 4 );
		double maxDev = 0.0;
		//bool isOK = lineFittingLS( straightString[i], parameters, maxDev );
		lineFittingSVD(&straightString[i][0], straightString[i].size(), parameters, maxDev);
		//if ( isOK )
		{
			double k = parameters[1];
			double b = parameters[2];
			int lineLen = straightString[i].size();

			double xs = 0, ys = 0, xe = 0, ye = 0;
			if ( ! parameters[0] )  // horizontal
			{
				xs = straightString[i][0].x;
				ys = k * xs + b;
				xe = straightString[i][lineLen-1].x;
				ye = k * xe + b;
			}
			else   // vertical
			{
				ys = straightString[i][0].y;
				xs = k * ys + b;
				ye = straightString[i][lineLen-1].y;
				xe = k * ye + b;
			}

			if ( !( xs==xe && ys==ye ) )
			{
				std::vector<cv::Point2d> lineCur(2);
				lineCur[0] = cv::Point2d(xs, ys);
				lineCur[1] = cv::Point2d(xe, ye);

				lines.push_back( lineCur );
			}
		}
	}
}

void LineFunctions::subDivision( std::vector<std::vector<cv::Point> > &straightString, std::vector<cv::Point> &contour, int first_index, int last_index
	, double min_deviation, int min_size )
{
	int clusters_count = straightString.size();

	cv::Point first = contour[first_index];
	cv::Point last = contour[last_index];

	// Compute the length of the straight line segment defined by the endpoints of the cluster.
	int x = first.x - last.x;
	int y = first.y - last.y;
	double length = sqrt( static_cast<double>( (x * x) + (y * y) ) );

	// Find the pixels with maximum deviation from the line segment in order to subdivide the cluster.
	int max_pixel_index = 0;
	double max_deviation = -1.0;

	for (int i=first_index, count=contour.size(); i!=last_index; i=(i+1)%count)
	{
		cv::Point current = contour[i];

		double deviation = static_cast<double>( abs( ((current.x - first.x) * (first.y - last.y)) + ((current.y - first.y) * (last.x - first.x)) ) );

		if (deviation > max_deviation)
		{
			max_pixel_index = i;
			max_deviation = deviation;
		}
	}
	max_deviation /= length;

	// 
	// 	// Compute the ratio between the length of the segment and the maximum deviation.
	// 	float ratio = length / std::max( max_deviation, min_deviation );

	// Test the number of pixels of the sub-clusters.
	int half_min_size=min_size/2;
	if ((max_deviation>=min_deviation) && ((max_pixel_index - first_index + 1) >= half_min_size) && ((last_index - max_pixel_index + 1) >= half_min_size))
	{
		subDivision( straightString, contour, first_index, max_pixel_index, min_deviation, min_size );
		subDivision( straightString, contour, max_pixel_index, last_index, min_deviation, min_size );
	}
	else
	{
		// 
		if ( last_index - first_index > min_size )
		{
			std::vector<cv::Point> straightStringCur;
			for ( int i=first_index; i<last_index; ++i )
			{
				straightStringCur.push_back(contour[i]);
			}
			straightString.push_back(straightStringCur);
			//terminalIds.push_back(std::pair<int,int>(first_index, last_index));
		}
	}
}

void LineFunctions::lineFittingSVD(cv::Point *points, int length, std::vector<double> &parameters, double &maxDev)
{
	// 
	cv::Matx21d h_mean( 0, 0 );
	for( int i = 0; i < length; ++i )
	{
		h_mean += cv::Matx21d( points[i].x, points[i].y );
	}
	h_mean *= ( 1.0 / length );

	cv::Matx22d h_cov( 0, 0, 0, 0 );
	for( int i = 0; i < length; ++i )
	{
		cv::Matx21d hi = cv::Matx21d( points[i].x, points[i].y );
		h_cov += ( hi - h_mean ) * ( hi - h_mean ).t();
	}
	h_cov *=( 1.0 / length );

	// eigenvector
	cv::Matx22d h_cov_evectors;
	cv::Matx21d h_cov_evals;
	cv::eigen( h_cov, h_cov_evals, h_cov_evectors );

	cv::Matx21d normal = h_cov_evectors.row(1).t();

	// 
	if ( abs(normal.val[0]) < abs(normal.val[1]) )  // horizontal
	{
		parameters[0] = 0;
		parameters[1] = - normal.val[0] / normal.val[1];
		parameters[2] = h_mean.val[1] - parameters[1] * h_mean.val[0];
	}
	else  // vertical
	{
		parameters[0] = 1;
		parameters[1] = - normal.val[1] / normal.val[0];
		parameters[2] = h_mean.val[0] - parameters[1] * h_mean.val[1];
	}

	// maximal deviation
	maxDev = 0;
	for( int i = 0; i < length; ++i )
	{
		cv::Matx21d hi = cv::Matx21d( points[i].x, points[i].y );
		cv::Matx21d v = hi - h_mean;
		double dis2 = v.dot(v);
		double disNormal = v.dot(normal);
		double disOrtho = sqrt(dis2 - disNormal*disNormal);
		if ( disOrtho > maxDev )
		{
			maxDev = disOrtho;
		}
	}
}

/************************************************************************/
/*                            PCA Functions                             */
/************************************************************************/


void PCAFunctions::Ori_PCA( PointCloud<double> &cloud, int k, std::vector<PCAInfo> &pcaInfos, double &scale, double &magnitd )
{
	double MINVALUE = 1e-7;
	int pointNum = cloud.pts.size();

	// 1. build kd-tree
	typedef KDTreeSingleIndexAdaptor< L2_Simple_Adaptor<double, PointCloud<double> >, PointCloud<double>, 3/*dim*/ > my_kd_tree_t;
	my_kd_tree_t index(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */) );
	index.buildIndex();

	// 2. knn search
	size_t *out_ks = new size_t[pointNum];
	size_t **out_indices = new size_t *[pointNum];
#pragma omp parallel for
	for (int i=0; i<pointNum; ++i)
	{
		double *query_pt = new double[3];
		query_pt[0] = cloud.pts[i].x;  query_pt[1] = cloud.pts[i].y;  query_pt[2] = cloud.pts[i].z;
		double *dis_temp = new double[k];
		out_indices[i]= new size_t[k];

		nanoflann::KNNResultSet<double> resultSet(k);
		resultSet.init(out_indices[i], dis_temp );
		index.findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));
		out_ks[i] = resultSet.size();

		delete query_pt;
		delete dis_temp;
	}
	index.freeIndex(index);

	// 3. PCA normal estimation
	scale = 0.0;
	pcaInfos.resize( pointNum );
#pragma omp parallel for
	for ( int i = 0; i < pointNum; ++i ) 
	{
		// 
		int ki = out_ks[i];

		double h_mean_x = 0.0, h_mean_y = 0.0, h_mean_z = 0.0;
		for( int j = 0; j < ki; ++j )
		{
			int idx = out_indices[i][j];
			h_mean_x += cloud.pts[idx].x;
			h_mean_y += cloud.pts[idx].y;
			h_mean_z += cloud.pts[idx].z;
		}
		h_mean_x *= 1.0/ki;  h_mean_y *= 1.0/ki; h_mean_z *= 1.0/ki;

		double h_cov_1 = 0.0, h_cov_2 = 0.0, h_cov_3 = 0.0;
		double h_cov_5 = 0.0, h_cov_6 = 0.0;
		double h_cov_9 = 0.0;
		double dx = 0.0, dy = 0.0, dz = 0.0;
		for( int j = 0; j < k; ++j )
		{
			int idx = out_indices[i][j];
			dx = cloud.pts[idx].x - h_mean_x;
			dy = cloud.pts[idx].y - h_mean_y;
			dz = cloud.pts[idx].z - h_mean_z;

			h_cov_1 += dx*dx; h_cov_2 += dx*dy; h_cov_3 += dx*dz;
			h_cov_5 += dy*dy; h_cov_6 += dy*dz;
			h_cov_9 += dz*dz;
		}
		cv::Matx33d h_cov(
			h_cov_1, h_cov_2, h_cov_3, 
			h_cov_2, h_cov_5, h_cov_6, 
			h_cov_3, h_cov_6, h_cov_9);
		h_cov *= 1.0/ki;

		// eigenvector
		cv::Matx33d h_cov_evectors;
		cv::Matx31d h_cov_evals;
		cv::eigen( h_cov, h_cov_evals, h_cov_evectors );

		// 
		pcaInfos[i].idxAll.resize( ki );
		for ( int j =0; j<ki; ++j )
		{
			int idx = out_indices[i][j];
			pcaInfos[i].idxAll[j] = idx;
		}

		int idx = out_indices[i][3];
		dx = cloud.pts[idx].x - cloud.pts[i].x;
		dy = cloud.pts[idx].y - cloud.pts[i].y;
		dz = cloud.pts[idx].z - cloud.pts[i].z;
		double scaleTemp = sqrt(dx*dx + dy*dy + dz*dz);
		pcaInfos[i].scale = scaleTemp;
		scale += scaleTemp;

		//pcaInfos[i].lambda0 = h_cov_evals.row(2).val[0];
		double t = h_cov_evals.row(0).val[0] + h_cov_evals.row(1).val[0] + h_cov_evals.row(2).val[0] + ( rand()%10 + 1 ) * MINVALUE;
		pcaInfos[i].lambda0 = h_cov_evals.row(2).val[0] / t;
		pcaInfos[i].normal = h_cov_evectors.row(2).t();

		// outliers removal via MCMD
		pcaInfos[i].idxIn = pcaInfos[i].idxAll;

		delete out_indices[i];
	}
	delete []out_indices;
	delete out_ks;

	scale /= pointNum;
	magnitd = sqrt(cloud.pts[0].x*cloud.pts[0].x + cloud.pts[0].y*cloud.pts[0].y + cloud.pts[0].z*cloud.pts[0].z);
}


void PCAFunctions::PCASingle( std::vector<std::vector<double> > &pointData, PCAInfo &pcaInfo )
{
	int i, j;
	int k = pointData.size();
	double a = 1.4826;
	double thRz = 2.5;

	// 
	pcaInfo.idxIn.resize( k );
	cv::Matx31d h_mean( 0, 0, 0 );
	for( i = 0; i < k; ++i )
	{
		pcaInfo.idxIn[i] = i;
		h_mean += cv::Matx31d( pointData[i][0], pointData[i][1], pointData[i][2] );
	}
	h_mean *= ( 1.0 / k );

	cv::Matx33d h_cov( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
	for( i = 0; i < k; ++i )
	{
		cv::Matx31d hi = cv::Matx31d( pointData[i][0], pointData[i][1], pointData[i][2] );
		h_cov += ( hi - h_mean ) * ( hi - h_mean ).t();
	}
	h_cov *=( 1.0 / k );

	// eigenvector
	cv::Matx33d h_cov_evectors;
	cv::Matx31d h_cov_evals;
	cv::eigen( h_cov, h_cov_evals, h_cov_evectors );

	// 
	pcaInfo.idxAll = pcaInfo.idxIn;
	//pcaInfo.lambda0 = h_cov_evals.row(2).val[0];
	pcaInfo.lambda0 = h_cov_evals.row(2).val[0] / ( h_cov_evals.row(0).val[0] + h_cov_evals.row(1).val[0] + h_cov_evals.row(2).val[0] );
	pcaInfo.normal  = h_cov_evectors.row(2).t();
	pcaInfo.planePt = h_mean;

	// outliers removal via MCMD
	MCMD_OutlierRemoval( pointData, pcaInfo );	
}

void PCAFunctions::MCMD_OutlierRemoval( std::vector<std::vector<double> > &pointData, PCAInfo &pcaInfo )
{
	double a = 1.4826;
	double thRz = 2.5;
	int num = pcaInfo.idxAll.size();

	// ODs
	cv::Matx31d h_mean( 0, 0, 0 );
	for( int j = 0; j < pcaInfo.idxIn.size(); ++j )
	{
		int idx = pcaInfo.idxIn[j];
		h_mean += cv::Matx31d( pointData[idx][0], pointData[idx][1], pointData[idx][2] );
	}
	h_mean *= ( 1.0 / pcaInfo.idxIn.size() );

	std::vector<double> ODs( num );
	for( int j = 0; j < num; ++j )
	{
		int idx = pcaInfo.idxAll[j];
		cv::Matx31d pt( pointData[idx][0], pointData[idx][1], pointData[idx][2] );
		cv::Matx<double, 1, 1> OD_mat = ( pt - h_mean ).t() * pcaInfo.normal;
		double OD = fabs( OD_mat.val[0] );
		ODs[j] = OD;
	}

	// calculate the Rz-score for all points using ODs
	std::vector<double> sorted_ODs( ODs.begin(), ODs.end() );
	double median_OD = meadian( sorted_ODs );
	std::vector<double>().swap( sorted_ODs );

	std::vector<double> abs_diff_ODs( num );
	for( int j = 0; j < num; ++j )
	{
		abs_diff_ODs[j] = fabs( ODs[j] - median_OD );
	}
	double MAD_OD = a * meadian( abs_diff_ODs ) + 1e-6;
	std::vector<double>().swap( abs_diff_ODs );

	// get inlier 
	std::vector<int> idxInlier;
	for( int j = 0; j < num; ++j )
	{
		double Rzi = fabs( ODs[j] - median_OD ) / MAD_OD;
		if ( Rzi < thRz ) 
		{
			int idx = pcaInfo.idxAll[j];
			idxInlier.push_back( idx );
		}
	}

	// 
	pcaInfo.idxIn = idxInlier;
} 


double PCAFunctions::meadian( std::vector<double> dataset )
{
	std::sort( dataset.begin(), dataset.end(), []( const double& lhs, const double& rhs ){ return lhs < rhs; } );
	if(dataset.size()%2 == 0)
	{
		return dataset[dataset.size()/2];
	}
	else
	{
		return (dataset[dataset.size()/2] + dataset[dataset.size()/2 + 1])/2.0;
	}
}
