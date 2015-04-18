/***********************************************************
     Starter code for Assignment 3

     This code was originally written by Jack Wang for
		    CSC418, SPRING 2005

		implements scene_object.h

***********************************************************/

#include <cmath>
#include <limits>
#include <iostream>
#include "scene_object.h"

bool UnitSquare::intersect( Ray3D& ray, const Matrix4x4& worldToModel,
		const Matrix4x4& modelToWorld ) {
	// TODO: implement intersection code for UnitSquare, which is
	// defined on the xy-plane, with vertices (0.5, 0.5, 0), 
	// (-0.5, 0.5, 0), (-0.5, -0.5, 0), (0.5, -0.5, 0), and normal
	// (0, 0, 1).
	//
	// Your goal here is to fill ray.intersection with correct values
	// should an intersection occur.  This includes intersection.point, 
	// intersection.normal, intersection.none, intersection.t_hit.   
	//
	// HINT: Remember to first transform the ray into object space  
	// to simplify the intersection test.
    
    Ray3D modelRay;                     // To store transformed model space ray.
    Vector3D surfaceNormal(0, 0, 1);    // Constant surface normal.
    double t;                           // Intersection parameter.
    
    // Transform ray to model space.
    modelRay.origin = worldToModel * ray.origin;
    modelRay.dir = worldToModel * ray.dir;;

    // Compute intersection between ray and XY-plane.
    double d_dot_n = modelRay.dir.dot(surfaceNormal);
  
    if (d_dot_n == 0.0) {
        // Ray is parallel to plane.
        return false;
    }
    
    // Compute intersection and check if it occurs infront or behind the camera.
    t = -(Vector3D(modelRay.origin[0], modelRay.origin[1], modelRay.origin[2]).dot(surfaceNormal)) / d_dot_n;
    if (t < ray.intersection.t_min || t > ray.intersection.t_max) {
        return false;
    }
    
    if (!ray.intersection.none && t > ray.intersection.t_hit) {
        return false;
    }
    
    // Compute intersection point and check againt unit square bounds.
    Point3D intPoint = modelRay.origin + (t * modelRay.dir);
    if (!(intPoint[0] >= -0.5 && intPoint[0] <= 0.5 &&
          intPoint[1] >= -0.5 && intPoint[1] <= 0.5)) {
        return false;
    }
    
    // Transform and set intersection info.
    ray.intersection.point = modelToWorld * intPoint;
    ray.intersection.normal = transNorm(worldToModel, surfaceNormal);
    ray.intersection.normal.normalize();
    ray.intersection.t_hit = t;
    ray.intersection.none = false;
    ray.intersection.inside = false;
    
	return true;
}

bool UnitSphere::intersect( Ray3D& ray, const Matrix4x4& worldToModel,
		const Matrix4x4& modelToWorld ) {
	// TODO: implement intersection code for UnitSphere, which is centred 
	// on the origin.  
	//
	// Your goal here is to fill ray.intersection with correct values
	// should an intersection occur.  This includes intersection.point, 
	// intersection.normal, intersection.none, intersection.t_hit.   
	//
	// HINT: Remember to first transform the ray into object space  
	// to simplify the intersection test.

    Ray3D modelRay;
    double discriminant;
    double a, b, c;
    double t, t0, t1;
    int inside = 1;
    
    // Transform ray to model space.
    modelRay.origin = worldToModel * ray.origin;
    modelRay.dir = worldToModel * ray.dir;
    
    // Compute coefficients of quadractic formula.
    Vector3D vOrigin(modelRay.origin[0], modelRay.origin[1], modelRay.origin[2]);
    a = modelRay.dir.dot(modelRay.dir);
    b = modelRay.dir.dot(vOrigin);
    c = vOrigin.dot(vOrigin) - 1.0;
    
    // Compute discriminant of quadractic formula.
    discriminant = (b * b) - (a * c);
  
    if (discriminant < 0) {
        // No real roots. No intersection.
        return false;
    }
    
    t0 = (-b - sqrt(discriminant)) / a;
    t1 = (-b + sqrt(discriminant)) / a;
    
    t = t0;
    if (t < ray.intersection.t_min || t > ray.intersection.t_max) {
        t = t1;
        if (t < ray.intersection.t_min || t > ray.intersection.t_max) {
            return false;
        }
        inside = -1;
        ray.intersection.inside = true;
    }
    
    if (!ray.intersection.none && t > ray.intersection.t_hit) {
        return false;
    }
  
    Point3D intPoint = modelRay.origin + (t * modelRay.dir);
    Vector3D surfaceNormal = inside * (intPoint - Point3D(0, 0, 0));
    
    ray.intersection.point = modelToWorld * intPoint;
    ray.intersection.normal = transNorm(worldToModel, surfaceNormal);
    ray.intersection.normal.normalize();
    ray.intersection.t_hit = t;
    ray.intersection.none = false;
    
    return true;
}


bool UnitCylinder::intersect( Ray3D& ray, const Matrix4x4& worldToModel,
        const Matrix4x4& modelToWorld ) {
 
    Ray3D modelRay;
    Point3D intPoint;
    Vector3D surfaceNormal;
    double discriminant;
    double a, b, c;
    double t0, t1, h, r, p_x, p_z, d_x, d_z;
    double t;
    bool isParallel = false;
   
    // 0 - cylinder_t0, 1 - cylinder_t1, 2 - top, 3 - bot
    double intersection_ts[4];
    double inf = std::numeric_limits<double>::infinity();
    
    // initialize intersection_ts to infinities
    for (int i=0; i < 4; i++) {
	    intersection_ts[i] = inf;
    }
    
    h = 1.0;
    r = 0.5;
   
    // Transform ray to model space.
    modelRay.origin = worldToModel * ray.origin;
    modelRay.dir = worldToModel * ray.dir;

    // Compute coefficients of quadractic formula.
    p_x = modelRay.origin[0];
    p_z = modelRay.origin[2];
   
    d_x = modelRay.dir[0];
    d_z = modelRay.dir[2];
   
    a = pow(d_x, 2) + pow(d_z, 2);
    b = (2 * p_x * d_x) + (2 * p_z * d_z);
    c = pow(p_x, 2) + pow(p_z, 2) - pow(r, 2);
   
    // Compute discriminant of quadractic formula.
    discriminant = (b * b) - (4 * a * c);

    if (discriminant >= 0) {

	    t0 = (-b - sqrt(discriminant)) / (2 * a);
	    t1 = (-b + sqrt(discriminant)) / (2 * a);
	    Point3D intPoint_0 = modelRay.origin + (t0 * modelRay.dir);
	    Point3D intPoint_1 = modelRay.origin + (t1 * modelRay.dir);

        if (t0 >= ray.intersection.t_min && t0 <= ray.intersection.t_max) {
            if ((intPoint_0[1] >= 0 && intPoint_0[1] <= h)) {
 
                intersection_ts[0] = t0;
            }
        }   

        if (t1 >= ray.intersection.t_min && t1 <= ray.intersection.t_max) {
            if ((intPoint_1[1] >= 0 && intPoint_1[1] <= h)) {
   
            intersection_ts[1] = t1;
            }
        }

        ray.intersection.inside = true;
    }

    // Check if it intersected the top or bottom cap
    Vector3D surfaceNormalTopCirc(0, 1, 0);    // Constant surface normal.
    Vector3D surfaceNormalBotCirc(0, -1, 0);   // Constant surface normal.
    double t_top, t_bot;                       // Intersection parameter.
   
    // Compute intersection between ray and ZX-plane.
    double d_dot_n_top = modelRay.dir.dot(surfaceNormalTopCirc);
    double d_dot_n_bot = modelRay.dir.dot(surfaceNormalBotCirc);

    if (d_dot_n_top == 0.0) {
  
        // Ray is parallel to planes
        isParallel = true;
    } 

    if (!isParallel) {
      
        Vector3D vOrigin(modelRay.origin[0], modelRay.origin[1], modelRay.origin[2]);
      
        t_top = -(vOrigin.dot(surfaceNormalTopCirc) - h) / d_dot_n_top;
        t_bot = -(vOrigin.dot(surfaceNormalBotCirc)) / d_dot_n_bot;
      
        Point3D intPoint_top = modelRay.origin + (t_top * modelRay.dir);
        Point3D intPoint_bot = modelRay.origin + (t_bot * modelRay.dir);
      
        if ((t_top >= ray.intersection.t_min && t_top <= ray.intersection.t_max) &&
            ((pow(intPoint_top[0], 2) + pow(intPoint_top[2], 2) - pow(r, 2)) <= 0)) {
   
            intersection_ts[2] = t_top;
        }
      
        if ((t_bot >= ray.intersection.t_min && t_bot <= ray.intersection.t_max) &&
            ((pow(intPoint_bot[0], 2) + pow(intPoint_bot[2], 2) - pow(r, 2)) <= 0)) {
 
            intersection_ts[3] = t_bot;
        }
    }

    // Get the smallest t value index
    int indexOfSmallest = 0;
    double smallest_t = intersection_ts[0];

    for (int i=0; i < 4; i++) {
        if (intersection_ts[i] < smallest_t) {
            smallest_t = intersection_ts[i];
            indexOfSmallest = i;
        }
    }
    
    if (smallest_t == inf) {
        return false;
    } else {
        t = smallest_t;

        if (!ray.intersection.none && t > ray.intersection.t_hit) {
            return false;
        }

        intPoint = modelRay.origin + (t * modelRay.dir);

        if (indexOfSmallest == 0 || indexOfSmallest == 1) {
            surfaceNormal = intPoint - Point3D(0, intPoint[1], 0);
        } else if (indexOfSmallest == 2) {
            surfaceNormal = surfaceNormalTopCirc;
        } else {
            surfaceNormal = surfaceNormalBotCirc;
        }
        ray.intersection.point = modelToWorld * intPoint;
        ray.intersection.normal = transNorm(worldToModel, surfaceNormal);
        ray.intersection.normal.normalize();
        ray.intersection.t_hit = t;
        ray.intersection.none = false;
        return true;
    }
 
}
