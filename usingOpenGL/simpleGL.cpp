/*
  A very simple OpenGL program
*/

// include GL libraries
#include "GL/freeglut.h"
#include "GL/gl.h"

// Display function
void draw()
{
	glClearColor(0,0,0,0);
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1,1,1);
	glOrtho(-1,1,-1,1,-1,1);
	glBegin(GL_POLYGON);
		glVertex2f(-0.5,-0.5);
		glVertex2f(-0.5, 0.5);
		glVertex2f( 0.5, 0.5);
		glVertex2f( 0.5,-0.5);
	glEnd();
	glFlush();
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE);
	glutInitWindowSize(500,500);
	glutInitWindowPosition(100,100);
	glutCreateWindow("OpenGL - Test include libraries");
	glutDisplayFunc(draw);
	
	glutMainLoop();

	return 0;
}
