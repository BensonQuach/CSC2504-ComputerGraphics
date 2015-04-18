/***********************************************************
     Starter code for Assignment 3

     This code was originally written by Jack Wang for
		    CSC418, SPRING 2005

		implements light_source.h

***********************************************************/

#include <cmath>
#include <cstdlib>
#include "light_source.h"


// Point Light Source Implementation:

Colour PointLight::shade( Ray3D& ray ) {
	// TODO: implement this function to fill in values for ray.col 
	// using phong shading.  Make sure your vectors are normalized, and
	// clamp colour values to 1.0.
	//
	// It is assumed at this point that the intersection information in ray 
	// is available.  So be sure that traverseScene() is called on the ray 
	// before this function.  

    Material* mat = ray.intersection.mat;
    
    // Normal is assumed to be normalized.
    Vector3D normal = ray.intersection.normal;
    normal.normalize();

    Vector3D lightDir = _pos - ray.intersection.point;
    lightDir.normalize();
    
    Vector3D viewDir = ray.origin - ray.intersection.point;
    viewDir.normalize();


    // Computing ambient shading.
    Colour ambientShade = mat->ambient * _col_ambient;

    // Computing diffuse shading.
    double lightAngle = std::max(0.0,  lightDir.dot(normal));
    Colour diffuseShade = lightAngle * (_col_diffuse * mat->diffuse);

    // Computing specular shading.
    Vector3D r = ((2 * (lightDir.dot(normal))) * normal) - lightDir;
    double specIntensity = pow(std::max(0.0, r.dot(viewDir)), mat->specular_exp);
    Colour specularShade = specIntensity * (mat->specular * _col_specular);
    
    return ambientShade + diffuseShade + specularShade;
}


Colour PointLight::ambientShade( Ray3D& ray ) {
    // Shades the given ray with ambient shading only. 
    return ray.intersection.mat->ambient * _col_ambient;
}


// Area Light Source Implementation:

Point3D AreaLight::get_position() const {
    // Return a sample point on the planer surface that this
    // AreaLight occupies. Then, the raytracer can repeatedly call
    // this function to sample points for soft shadows.   
    double rand1 = (double)rand() / (double)RAND_MAX;
    double rand2 = (double)rand() / (double)RAND_MAX;
    return _corner + rand1*_axis1 + rand2*_axis2;
}

Colour AreaLight::shade( Ray3D& ray ) {
    // Arealight shading routine. The shading routine for area lights is the same
    // as for point lights except we dont accumulate the ambient shade here. The
    // raytracer must add ambient light after computing the average of the specular
    // and diffuse lighting components.
    //
    // Also, color values are not clamped here.

    Material* mat = ray.intersection.mat;
    
    // Normal is assumed to be normalized.
    Vector3D normal = ray.intersection.normal;
    normal.normalize();

    Vector3D lightDir = _pos - ray.intersection.point;
    lightDir.normalize();
    
    Vector3D viewDir = ray.origin - ray.intersection.point;
    viewDir.normalize();

    // Compute diffuse component.
    double lightAngle = std::max(0.0,  lightDir.dot(normal));
    Colour diffuseShade = lightAngle * (_col_diffuse * mat->diffuse);

    // Compute specular component.
    Vector3D r = ((2 * (lightDir.dot(normal))) * normal) - lightDir;
    double specIntensity = pow(std::max(0.0, r.dot(viewDir)), mat->specular_exp);
    Colour specularShade = specIntensity * (mat->specular * _col_specular);
    
    return diffuseShade + specularShade;
}

Colour AreaLight::ambientShade( Ray3D& ray) {
    // Shades the given ray with ambient shading only.
    return ray.intersection.mat->ambient * _col_ambient;
}
  


