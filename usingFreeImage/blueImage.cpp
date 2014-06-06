/*
Code from 
lovehateubuntu.blogspot.co.uk/2009/06/using-freeimage-in-ubuntu.html
Simple code that uses the freeimage library to output a blue image to file
*/

#include <FreeImage.h>
#include <stdlib.h>

int main()
{
	FreeImage_Initialise();
	atexit(FreeImage_DeInitialise);

	// Create the bitmap object
	FIBITMAP * bitmap = FreeImage_Allocate(200, 200, 32);

	// Create the blue colour
	RGBQUAD blue;
	blue.rgbBlue = 255;

	for(int i=0;i<200;i++)
	{
		for(int j=0;j<200;j++)
		{
			FreeImage_SetPixelColor(bitmap, i, j, &blue);
		}
	}

	// Save it as .bmp
	FreeImage_Save(FIF_BMP, bitmap, "output.bmp");
	
	// Deallocate memory
	FreeImage_Unload(bitmap);
}
