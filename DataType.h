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
	//�ļ����  
	intptr_t   hFile = 0;
	std::string separator = "\\*txt";

	std::string filename = path + separator;
	const char* searchfile = filename.data();
	//�ļ���Ϣ  
	struct _finddata_t fileinfo;  //�ļ���Ϣ��ȡ�ṹ

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
	//�ļ����  
	intptr_t   hFile = 0;
	std::string separator = "\\*bmp";
	
	std::string filename = path + separator ;
	const char* searchfile = filename.data();
	//�ļ���Ϣ  
	struct _finddata_t fileinfo;  //�ļ���Ϣ��ȡ�ṹ
	
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

