#pragma once
#include <opencv2\opencv.hpp>
#include <string>
#include <vector>
#include "DataType.h"

#define PI 3.1415926
#define CALIBPANEL_SPACING_X 35.0
#define CALIBPANEL_SPACING_Y 35.0
#define CALIBPANEL_NUMBERS_X 9
#define CALIBPANEL_NUMBERS_Y 11


static bool circularityCriteria1(double perimeter, double area)
{
	double c = perimeter * perimeter / abs(area);
	double factor = c / (4 * PI);
	if (perimeter > 1500)
	{
		return false;
	}
	else if (factor >= 1.0 && factor <= 1.5)
	{
		return true;
	}
	else 
	{
		return false;
	}
		
}

static void pointVecFromTwoPoints(cv::Point2d p1, cv::Point2d p2, cv::Point2d& direction)
{
	direction = cv::Point2d(p2.x - p1.x, p2.y - p1.y);
}

class CalibPanelDetector
{
public:
	CalibPanelDetector();
	CalibPanelDetector(std::string path);
	~CalibPanelDetector();
	void EntryEllipseDetector();
	void Processing();
	void Build3DPointsForCalibPanel();
	void HomographyFromFivePoints();
	void ShowResult();
	double computer_distance(cv::Point2f p1, cv::Point2f p2);
	void SaveResult(std::string& path);
	std::vector<MatchPoint> getMatchPoint() const;

private:
	cv::Mat _SrcImage;
	cv::Mat _ShowImage;
	cv::Mat _vCalibPanel3DPoints;
	std::vector<cv::RotatedRect> _vEllipseBoxs;
	double _AverageLength;
	std::vector<cv::Point2d> _FiveBigCenter;
	std::vector<MatchPoint> _vCalibPanelMatchPoints;



};



