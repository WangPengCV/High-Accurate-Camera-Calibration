#include "CalibPanelDetector.h"
#include <string>
#include <vector>
#include "DataType.h"
#include "InitialValueEstimate.h"
#include "BundleAdjustment.h"
#include <fstream>

void readResult(const std::string& filename,cv::Mat& KL,cv::Mat& distCoeffsL,cv::Mat& kR, cv::Mat& distCoeffsR,
	cv::Mat& Left2RightRotate,cv::Mat& Left2RightTranslate)
{
	cv::FileStorage fr(filename, cv::FileStorage::READ);
	cv::FileNode n = fr["LeftCamera"];
	cv::Mat LeftcameraMatrix(3, 3, CV_64F,cv::Scalar(0)),LeftdistCoeffs(5,1,CV_64F,cv::Scalar(0));
	cv::FileNode fx = n["fx"];
	cv::FileNode fy = n["fy"];
	cv::FileNode cx = n["cx"];
	cv::FileNode cy = n["cy"];
	cv::FileNode k1 = n["k1"];
	cv::FileNode k2 = n["k2"];
	cv::FileNode p1 = n["p1"];
	cv::FileNode p2 = n["p2"];
	LeftcameraMatrix.at<double>(0, 0) = fx;
	LeftcameraMatrix.at<double>(1, 1) = fy;
	LeftcameraMatrix.at<double>(0, 2) = cx;
	LeftcameraMatrix.at<double>(1, 2) = cy;
	LeftcameraMatrix.at<double>(2, 2) = 1;
	LeftdistCoeffs.at<double>(0, 0) = k1;
	LeftdistCoeffs.at<double>(1, 0) = k2;
	LeftdistCoeffs.at<double>(2, 0) = p1;
	LeftdistCoeffs.at<double>(3, 0) = p2;
	KL = LeftcameraMatrix;
	distCoeffsL = LeftdistCoeffs;

	n = fr["RightCamera"];
	cv::Mat RightcameraMatrix(3, 3, CV_64F, cv::Scalar(0)), RightdistCoeffs(5, 1, CV_64F, cv::Scalar(0));
	fx = n["fx"];
	fy = n["fy"];
	cx = n["cx"];
	cy = n["cy"];
	k1 = n["k1"];
	k2 = n["k2"];
	p1 = n["p1"];
	p2 = n["p2"];
	RightcameraMatrix.at<double>(0, 0) = fx;
	RightcameraMatrix.at<double>(1, 1) = fy;
	RightcameraMatrix.at<double>(0, 2) = cx;
	RightcameraMatrix.at<double>(1, 2) = cy;
	RightcameraMatrix.at<double>(2, 2) = 1;
	RightdistCoeffs.at<double>(0, 0) = k1;
	RightdistCoeffs.at<double>(1, 0) = k2;
	RightdistCoeffs.at<double>(2, 0) = p1;
	RightdistCoeffs.at<double>(3, 0) = p2;
	kR = RightcameraMatrix;
	distCoeffsR = RightdistCoeffs;

	n = fr["Left2Right"];
	cv::Mat Rotate(3, 1, CV_64F, cv::Scalar(0)), Translate(3, 1, CV_64F, cv::Scalar(0));
	cv::FileNode rotate = n["rotate"];
	cv::FileNode translate = n["translate"];
	Rotate.at<double>(0, 0) = rotate[0];
	Rotate.at<double>(1, 0) = rotate[1];
	Rotate.at<double>(2, 0) = rotate[2];
	Translate.at<double>(0, 0) = translate[3];
	Translate.at<double>(1, 0) = translate[4];
	Translate.at<double>(2, 0) = translate[5];
	Left2RightRotate = Rotate;
	Left2RightTranslate = Translate;
}

int main()
{

	
	std::string imageName = "Right";
	std::vector<std::string> fileNames;
	getAllFiles_bmp(imageName, fileNames);
	std::vector<std::vector<MatchPoint>> RCalibpanelPoint;
	RCalibpanelPoint.resize(fileNames.size(), std::vector<MatchPoint>());
	for (size_t i = 0; i < fileNames.size(); ++i)
	{
		CalibPanelDetector a("Right/" + fileNames[i]);
		a.EntryEllipseDetector();
		//a.ShowResult();
		RCalibpanelPoint[i] = a.getMatchPoint();
	}

	imageName = "Left";
	fileNames.clear();
	fileNames.clear();
	getAllFiles_bmp(imageName, fileNames);
	std::vector<std::vector<MatchPoint>> LCalibpanelPoint;
	LCalibpanelPoint.resize(fileNames.size(), std::vector<MatchPoint>());
	for (size_t i = 0; i < fileNames.size(); ++i)
	{
		CalibPanelDetector b("Left/" + fileNames[i]);
		b.EntryEllipseDetector();
		//b.ShowResult();
		LCalibpanelPoint[i] = b.getMatchPoint();
	}
	/*BundleAdjustment opotimiz(LCalibpanelPoint, RCalibpanelPoint);
	opotimiz.getInitialValue();
	opotimiz.Optimize();
	opotimiz.SaveResult("result.yml");*/

	//读取标定结果
	cv::Mat LeftcameraMatrix(3, 3, CV_64F, cv::Scalar(0)), LeftdistCoeffs(5, 1, CV_64F, cv::Scalar(0));
	cv::Mat RightcameraMatrix(3, 3, CV_64F, cv::Scalar(0)), RightdistCoeffs(5, 1, CV_64F, cv::Scalar(0));
	cv::Mat Rotate(3, 1, CV_64F, cv::Scalar(0)), Translate(3, 1, CV_64F, cv::Scalar(0));
	readResult("result.yml", LeftcameraMatrix, LeftdistCoeffs, RightcameraMatrix, RightdistCoeffs,Rotate,Translate);
	//重建标定板
	cv::Mat LeftCameraP, RightCameraP;
	cv::Mat LeftCameraR = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat LeftCameraT = cv::Mat::zeros(3, 1, CV_64F);
	cv::Mat LeftCameraRT; 
	cv::Mat RightCameraRt;
	cv::hconcat(LeftCameraR, LeftCameraT, LeftCameraRT);
	cv::Rodrigues(Rotate, Rotate);
	cv::hconcat(Rotate, Translate, RightCameraRt);
	LeftCameraP = LeftcameraMatrix* LeftCameraRT;
	RightCameraP = RightcameraMatrix * RightCameraRt;

	

	std::vector<double> D;
	for (int num = 0; num < LCalibpanelPoint.size(); ++num)
	{
		std::vector<cv::Point2d> Leftpoints;
		std::vector<cv::Point2d> Rightpoints;
		std::vector<MatchPoint> leftone = LCalibpanelPoint[num];
		std::vector<MatchPoint> rightone = RCalibpanelPoint[num];
		for (int i = 0; i < leftone.size(); i++)
		{
			/*if (leftone[i].identifier == 22 || leftone[i].identifier == 76)
			{*/
				cv::Point2d point_l(leftone[i].imgpoint.x, leftone[i].imgpoint.y);
				Leftpoints.push_back(point_l);
			/*}*/
			//if (leftone[i].identifier == 22 || leftone[i].identifier == 76)
			/*{*/
				cv::Point2d point_r(rightone[i].imgpoint.x, rightone[i].imgpoint.y);
				Rightpoints.push_back(point_r);
			/*}*/




		}
		//图像去畸变
		cv::undistortPoints(Leftpoints, Leftpoints, LeftcameraMatrix, LeftdistCoeffs, cv::noArray(), LeftcameraMatrix);
		cv::undistortPoints(Rightpoints, Rightpoints, RightcameraMatrix, RightdistCoeffs, cv::noArray(), RightcameraMatrix);

		//三角重建
		cv::Mat pnts4D;
		cv::triangulatePoints(LeftCameraP, RightCameraP, Leftpoints, Rightpoints, pnts4D);
		std::vector<cv::Point3d> points;
		for (int i = 0; i < pnts4D.cols; i++)
		{
			cv::Mat x = pnts4D.col(i);
			x /= x.at<double>(3, 0); // 归一化：此处的归一化是指从齐次坐标变换到非齐次坐标。而不是变换到归一化平面。
			cv::Point3d p(
				x.at<double>(0, 0),
				x.at<double>(1, 0),
				x.at<double>(2, 0)
			);
			points.push_back(p);
		}
		std::ofstream write;
		//note: the 3d coordinate is in left cameral coordinate,so we can use ralative distance to evaluate result
		write.open("POINT3D.txt", std::ios::app);
		for (int i = 0; i < points.size(); ++i)
		{
			write << points[i].x << " " << points[i].y << " " << points[i].z << std::endl;
		}
		write.close();
		cv::Vec3d vector22to76(points[22].x - points[76].x, points[22].y - points[76].y, points[22].z - points[76].z);
		double distance = cv::norm(vector22to76);
		D.push_back(distance);
		points.clear();
	}
	std::ofstream write;
	write.open("Distance3D.txt", std::ios::app);
	for (int i = 0; i < D.size(); ++i)
	{
		write << D[i] <<  std::endl;
	}
	write.close();
	
	
	//
	

	return 0;
}