//#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include "windows.h"
#include "dshow.h"
#include <iostream>
#include <string>
#include <vector>
#include <comutil.h>

#pragma comment(lib, "strmiids.lib")
#pragma comment(lib, "quartz.lib")
#pragma comment(lib, "comsuppw.lib")

//using namespace cv;
using namespace std;

int listVideoDevices(vector<string>& list) {

	//COM Library Initialization
	//comInit();

	ICreateDevEnum *pDevEnum = NULL;
	IEnumMoniker *pEnum = NULL;
	int deviceCounter = 0;
	CoInitialize(NULL);

	HRESULT hr = CoCreateInstance(CLSID_SystemDeviceEnum, NULL,
		CLSCTX_INPROC_SERVER, IID_ICreateDevEnum,
		reinterpret_cast<void**>(&pDevEnum));


	if (SUCCEEDED(hr))
	{
		// Create an enumerator for the video capture category.
		hr = pDevEnum->CreateClassEnumerator(
			CLSID_VideoInputDeviceCategory,
			&pEnum, 0);

		if (hr == S_OK) {

			// printf("SETUP: Looking For Capture Devices\n");
			IMoniker *pMoniker = NULL;

			while (pEnum->Next(1, &pMoniker, NULL) == S_OK) {

				IPropertyBag *pPropBag;
				hr = pMoniker->BindToStorage(0, 0, IID_IPropertyBag,
					(void**)(&pPropBag));

				if (FAILED(hr)) {
					pMoniker->Release();
					continue;  // Skip this one, maybe the next one will work.
				}
				VARIANT varName;  // Find the description or friendly name.
				VariantInit(&varName);
				hr = pPropBag->Read(L"Description", &varName, 0);

				if (FAILED(hr)) hr = pPropBag->Read(L"FriendlyName", &varName, 0);

				if (SUCCEEDED(hr))
				{
					hr = pPropBag->Read(L"FriendlyName", &varName, 0);
                    _bstr_t   str(varName.bstrVal);
					string s(str);
					list.push_back(s);
				}

				pPropBag->Release();
				pPropBag = NULL;

				pMoniker->Release();
				pMoniker = NULL;

				deviceCounter++;
			}

			pDevEnum->Release();
			pDevEnum = NULL;

			pEnum->Release();
			pEnum = NULL;
		}
	}
	return deviceCounter;
}
int listAudioDevices(vector<string>& list) {

	//COM Library Initialization
	//comInit();

	ICreateDevEnum *pDevEnum = NULL;
	IEnumMoniker *pEnum = NULL;
	int deviceCounter = 0;
	CoInitialize(NULL);

	HRESULT hr = CoCreateInstance(CLSID_SystemDeviceEnum, NULL,
		CLSCTX_INPROC_SERVER, IID_ICreateDevEnum,
		reinterpret_cast<void**>(&pDevEnum));


	if (SUCCEEDED(hr))
	{
		// Create an enumerator for the video capture category.
		hr = pDevEnum->CreateClassEnumerator(
			CLSID_AudioInputDeviceCategory,
			&pEnum, 0);

		if (hr == S_OK) {

			// printf("SETUP: Looking For Capture Devices\n");
			IMoniker *pMoniker = NULL;

			while (pEnum->Next(1, &pMoniker, NULL) == S_OK) {

				IPropertyBag *pPropBag;
				hr = pMoniker->BindToStorage(0, 0, IID_IPropertyBag,
					(void**)(&pPropBag));

				if (FAILED(hr)) {
					pMoniker->Release();
					continue;  // Skip this one, maybe the next one will work.
				}

				// Find the description or friendly name.
				VARIANT varName;
				VariantInit(&varName);
				hr = pPropBag->Read(L"Description", &varName, 0);

				if (FAILED(hr)) hr = pPropBag->Read(L"FriendlyName", &varName, 0);

				if (SUCCEEDED(hr))
				{

					hr = pPropBag->Read(L"FriendlyName", &varName, 0);

//					int count = 0;
//					char tmp[255] = { 0 };
//					//int maxLen = sizeof(deviceNames[0]) / sizeof(deviceNames[0][0]) - 2;
//					while (varName.bstrVal[count] != 0x00 && count < 255)
//					{
//						tmp[count] = (char)varName.bstrVal[count];
//						count++;
//					}
                    _bstr_t   str(varName.bstrVal);
					string s(str);
					list.push_back(s);
					//deviceNames[deviceCounter][count] = 0;

					//if (!silent) DebugPrintOut("SETUP: %i) %s\n", deviceCounter, deviceNames[deviceCounter]);
				}

				pPropBag->Release();
				pPropBag = NULL;

				pMoniker->Release();
				pMoniker = NULL;

				deviceCounter++;
			}

			pDevEnum->Release();
			pDevEnum = NULL;

			pEnum->Release();
			pEnum = NULL;
		}

		//if (!silent) DebugPrintOut("SETUP: %i Device(s) found\n\n", deviceCounter);
	}

	//comUnInit();

	return deviceCounter;
}

int test()
{
	vector<string> list;
	listVideoDevices(list);
	cout << "dev_size =      " << list.size() << endl;
	for (int i = 0; i < list.size(); i++)
	{
		cout << "device lists:" << list[i] << '\t' << "i=" << i << endl;
		// if (list[i] == "CY3014 USB, Analog 01 Capture")
	}
	// getchar();

	vector<string> audioList;
	listAudioDevices(audioList);
	cout << "audio_dev_size =      " << audioList.size() << endl;
	for (int i = 0; i < audioList.size(); i++)
	{
		cout << "device lists:" << list[i] << '\t' << "i=" << i << endl;
	}
	return 0;
}