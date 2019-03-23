#pragma once
#include <opencv2\opencv.hpp>
#include <io.h>

struct MatchPoint
{
	cv::Point2d imgpoint;
	cv::Point3d objpoint;
	int identifier;

};
struct CorrespondingPoint
{
	cv::Point2d imgpoint;
	cv::Point2d objpoint;
};


static void getAllFiles(std::string path, std::vector<std::string>& files)
{
	//文件句柄  
	intptr_t   hFile = 0;
	std::string separator = "\\*txt";

	std::string filename = path + separator;
	const char* searchfile = filename.data();
	//文件信息  
	struct _finddata_t fileinfo;  //文件信息读取结构

	hFile = _findfirst(searchfile, &fileinfo);
	if (hFile != -1)
	{
		files.push_back(fileinfo.name);
		while ((_findnext(hFile, &fileinfo) == 0))
		{
			files.push_back(fileinfo.name);
		}

	}
	_findclose(hFile);
}

static void getAllFiles_bmp(std::string path, std::vector<std::string>& files)
{
	//文件句柄  
	intptr_t   hFile = 0;
	std::string separator = "\\*bmp";
	
	std::string filename = path + separator ;
	const char* searchfile = filename.data();
	//文件信息  
	struct _finddata_t fileinfo;  //文件信息读取结构
	
	hFile =  _findfirst(searchfile, &fileinfo);
	if (hFile != -1)
	{
		files.push_back(fileinfo.name);
		while ((_findnext(hFile, &fileinfo) == 0))
		{
			files.push_back(fileinfo.name);
		}
		
	}
	_findclose(hFile);
	
}

