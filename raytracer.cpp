/***********************************************************
     Starter code for Assignment 3

     This code was originally written by Jack Wang for
            CSC418, SPRING 2005

        Implementations of functions in raytracer.h, 
        and the main function which specifies the 
        scene to be rendered.   

***********************************************************/

#include "raytracer.h"
#include "bmp_io.h"
#include <cmath>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <limits>

Raytracer::Raytracer() : _lightSource(NULL) {
    _root = new SceneDagNode();
}

Raytracer::~Raytracer() {
    delete _root;
}

SceneDagNode* Raytracer::addObject( SceneDagNode* parent, 
        SceneObject* obj, Material* mat ) {
    SceneDagNode* node = new SceneDagNode( obj, mat );
    node->parent = parent;
    node->next = NULL;
    node->child = NULL;
    
    // Add the object to the parent's child list, this means
    // whatever transformation applied to the parent will also
    // be applied to the child.
    if (parent->child == NULL) {
        parent->child = node;
    }
    else {
        parent = parent->child;
        while (parent->next != NULL) {
            parent = parent->next;
        }
        parent->next = node;
    }
    
    return node;;
}

LightListNode* Raytracer::addLightSource( LightSource* light ) {
    LightListNode* tmp = _lightSource;
    _lightSource = new LightListNode( light, tmp );
    return _lightSource;
}

void Raytracer::rotate( SceneDagNode* node, char axis, double angle ) {
    Matrix4x4 rotation;
    double toRadian = 2*M_PI/360.0;
    int i;
    
    for (i = 0; i < 2; i++) {
        switch(axis) {
            case 'x':
                rotation[0][0] = 1;
                rotation[1][1] = cos(angle*toRadian);
                rotation[1][2] = -sin(angle*toRadian);
                rotation[2][1] = sin(angle*toRadian);
                rotation[2][2] = cos(angle*toRadian);
                rotation[3][3] = 1;
            break;
            case 'y':
                rotation[0][0] = cos(angle*toRadian);
                rotation[0][2] = sin(angle*toRadian);
                rotation[1][1] = 1;
                rotation[2][0] = -sin(angle*toRadian);
                rotation[2][2] = cos(angle*toRadian);
                rotation[3][3] = 1;
            break;
            case 'z':
                rotation[0][0] = cos(angle*toRadian);
                rotation[0][1] = -sin(angle*toRadian);
                rotation[1][0] = sin(angle*toRadian);
                rotation[1][1] = cos(angle*toRadian);
                rotation[2][2] = 1;
                rotation[3][3] = 1;
            break;
        }
        if (i == 0) {
            node->trans = node->trans*rotation;     
            angle = -angle;
        } 
        else {
            node->invtrans = rotation*node->invtrans; 
        }   
    }
}

void Raytracer::translate( SceneDagNode* node, Vector3D trans ) {
    Matrix4x4 translation;
    
    translation[0][3] = trans[0];
    translation[1][3] = trans[1];
    translation[2][3] = trans[2];
    node->trans = node->trans*translation;  
    translation[0][3] = -trans[0];
    translation[1][3] = -trans[1];
    translation[2][3] = -trans[2];
    node->invtrans = translation*node->invtrans; 
}

void Raytracer::scale( SceneDagNode* node, Point3D origin, double factor[3] ) {
    Matrix4x4 scale;
    
    scale[0][0] = factor[0];
    scale[0][3] = origin[0] - factor[0] * origin[0];
    scale[1][1] = factor[1];
    scale[1][3] = origin[1] - factor[1] * origin[1];
    scale[2][2] = factor[2];
    scale[2][3] = origin[2] - factor[2] * origin[2];
    node->trans = node->trans*scale;    
    scale[0][0] = 1/factor[0];
    scale[0][3] = origin[0] - 1/factor[0] * origin[0];
    scale[1][1] = 1/factor[1];
    scale[1][3] = origin[1] - 1/factor[1] * origin[1];
    scale[2][2] = 1/factor[2];
    scale[2][3] = origin[2] - 1/factor[2] * origin[2];
    node->invtrans = scale*node->invtrans; 
}

Matrix4x4 Raytracer::initInvViewMatrix( Point3D eye, Vector3D view, 
        Vector3D up ) {
    Matrix4x4 mat; 
    Vector3D w;
    view.normalize();
    up = up - up.dot(view)*view;
    up.normalize();
    w = view.cross(up);

    mat[0][0] = w[0];
    mat[1][0] = w[1];
    mat[2][0] = w[2];
    mat[0][1] = up[0];
    mat[1][1] = up[1];
    mat[2][1] = up[2];
    mat[0][2] = -view[0];
    mat[1][2] = -view[1];
    mat[2][2] = -view[2];
    mat[0][3] = eye[0];
    mat[1][3] = eye[1];
    mat[2][3] = eye[2];

    return mat; 
}

void Raytracer::traverseScene( SceneDagNode* node, Ray3D& ray ) {
    SceneDagNode *childPtr;

    // Applies transformation of the current node to the global
    // transformation matrices.
    _modelToWorld = _modelToWorld*node->trans;
    _worldToModel = node->invtrans*_worldToModel; 
    if (node->obj) {
        // Perform intersection.
        if (node->obj->intersect(ray, _worldToModel, _modelToWorld)) {
            ray.intersection.mat = node->mat;
        }
    }
    // Traverse the children.
    childPtr = node->child;
    while (childPtr != NULL) {
        traverseScene(childPtr, ray);
        childPtr = childPtr->next;
    }

    // Removes transformation of the current node from the global
    // transformation matrices.
    _worldToModel = node->trans*_worldToModel;
    _modelToWorld = _modelToWorld*node->invtrans;
}

bool Raytracer::isRayVisible( Ray3D& ray, LightSource* light )
{
    // Trace a single shadow ray to check for occluding surfaces.
    Ray3D shadowRay( ray.intersection.point, light->get_position() - ray.intersection.point );
    shadowRay.intersection.t_min = M_EPSILON;
    shadowRay.intersection.t_max = 1;
    traverseScene(_root, shadowRay);   

    return shadowRay.intersection.none;
}


void Raytracer::computeShading( Ray3D& ray ) {
    
    LightListNode* curLight = _lightSource;
    LightSource* light;
    Colour shade(0,0,0);
    int nvisibleRays = 0;
    int nsamples = 30; // Number of samples for shading with area lights.

    for (;;) {
        if (curLight == NULL) break;
        light = curLight->light;
        nvisibleRays = 0;
    
        if ( light->get_type() == LightSource::POINT_LIGHT ) 
        {
            // Check for occluding geometry.
            if (isRayVisible(ray, light)) {
                shade = light->shade(ray);
            } else {
                shade = light->ambientShade(ray);
            }
        } 
        else if ( light->get_type() == LightSource::AREA_LIGHT ) 
        {
            // Repeatedly call isRayVisible. When isRayVisible call
            // light->get_position, the light will provide a random 
            // sample point.
            for (int i = 0; i < nsamples; i++) {
                if (isRayVisible(ray,light)) {
                    nvisibleRays++;
                }
            }
            shade = (double(nvisibleRays) / nsamples) * light->shade(ray);
            shade = shade + light->ambientShade(ray);
        }
        
        ray.col = ray.col + shade;
        curLight = curLight->next;
    }

    ray.col.clamp();
}


void Raytracer::initPixelBuffer() {
    int numbytes = _scrWidth * _scrHeight * sizeof(unsigned char);
    _rbuffer = new unsigned char[numbytes];
    _gbuffer = new unsigned char[numbytes];
    _bbuffer = new unsigned char[numbytes];
    for (int i = 0; i < _scrHeight; i++) {
        for (int j = 0; j < _scrWidth; j++) {
            _rbuffer[i*_scrWidth+j] = 0;
            _gbuffer[i*_scrWidth+j] = 0;
            _bbuffer[i*_scrWidth+j] = 0;
        }
    }
}

void Raytracer::flushPixelBuffer( char *file_name ) {
    bmp_write( file_name, _scrWidth, _scrHeight, _rbuffer, _gbuffer, _bbuffer );
    delete _rbuffer;
    delete _gbuffer;
    delete _bbuffer;
}


Vector3D Raytracer::reflect(Vector3D& incident, Vector3D& normal) {
    double cosI = -incident.dot(normal);
    return incident + (2*cosI)*normal;
}


bool Raytracer::refract(Vector3D& incident, Vector3D& normal, double n1, double n2, Vector3D* t) {
    double n = n1 / n2;
    double cosi = -(normal.dot(incident));
    double sint2 = n*n * (1.0 - cosi*cosi);
    if (sint2 > 1.0) {
        // Total internal relection.
        return false;
    }
    double cost = sqrt(1.0 - sint2);
    (*t) = (n*incident) + (n*cosi - cost)*normal;
    return true;
}

Colour Raytracer::shadeRay( Ray3D& ray , int depth, int maxDepth ) {

    Colour col(0.0, 0.0, 0.0);
    Colour localCol(0.0, 0.0, 0.0); 
    Colour globalCol(0.0, 0.0, 0.0);
    Vector3D normal, incident, r, t;
    Material* mat;

    traverseScene(_root, ray); 
    
    // Don't bother shading if the ray didn't hit anything.
    if (!ray.intersection.none) {
        // Compute local shading component.
        computeShading(ray); 
        localCol = ray.col;  
        col = localCol;

        // Compute global shading component.
        if (depth < maxDepth) {
            // Ray direction and normal are assumed to be normalized.
            incident = ray.dir;
            normal = ray.intersection.normal;
            mat = ray.intersection.mat;

            // Compute shading from reflected lights.
            if (mat->is_reflective()) {
                
                if (_glossy_reflections) {
                    // GLOSSY REFLECTIONS START ----------------
                
                    r = reflect(incident, normal);
                    r.normalize();
                
                    double samples, a, u, v; 
                    double min_component; 
                    double t_x, t_y, t_z;
                    double eps, eps_prime;
                    Vector3D r_prime, t_vect, w_vect, u_vect, v_vect;
                
                    // Parameters. Sample count and blur.
                    samples = 40;
                    a = 0.05; 
                
                    // Construct orthonormal basis on surface.
                    w_vect = r;
                    min_component = std::min(w_vect[0], std::min(w_vect[1], w_vect[2]));
                    t_x = (w_vect[0] == min_component) ? 1 : w_vect[0];
                    t_y = (w_vect[1] == min_component) ? 1 : w_vect[1];
                    t_z = (w_vect[2] == min_component) ? 1 : w_vect[2];
                    t_vect = Vector3D(t_x, t_y, t_z);
                    u_vect = cross(t_vect, w_vect);
                    u_vect.normalize();
                    v_vect = cross(w_vect, u_vect);


                    for (int k = 0; k < samples; k++) {
                    
                        eps = (double)rand() / (double)RAND_MAX;
                        eps_prime = (double)rand() / (double)RAND_MAX;
                    
                        u = -(a/2) + eps*a;
                        v = -(a/2) + eps_prime*a;
                    
                        r_prime = r + (u * u_vect) + (v * v_vect);
                        r_prime.normalize();
                
                        if (r_prime.dot(normal) >= 0) {
                            Ray3D reflectRay(ray.intersection.point, r_prime);
                            reflectRay.intersection.t_min = M_EPSILON;
                            reflectRay.intersection.t_max = std::numeric_limits<double>::infinity();
                            reflectRay.refract_index = ray.refract_index;

                            globalCol = globalCol + shadeRay(reflectRay, depth+1, maxDepth);
                        }
                    }
                    globalCol = (mat->reflect_cof / samples) * globalCol;
                
                    // GLOSSY REFLECTIONS END --------------------
                } else {
                    r = reflect(incident, normal);
                    
                    Ray3D reflectRay(ray.intersection.point, r);
                    reflectRay.intersection.t_min = M_EPSILON;
                    reflectRay.intersection.t_max = std::numeric_limits<double>::infinity();
                    reflectRay.refract_index = ray.refract_index;
                    
                    globalCol = globalCol + mat->reflect_cof * shadeRay(reflectRay, depth+1, maxDepth);
                }
            }

            // Compute shading of refracted light.
            if (mat->is_refractive()) {
                
                if (refract(incident, normal, ray.refract_index, mat->refract_index, &t)) {
                    
                    Ray3D refractRay(ray.intersection.point, t);
                    refractRay.intersection.t_min = M_EPSILON;
                    refractRay.intersection.t_max = std::numeric_limits<double>::infinity();
                    if (!ray.intersection.inside) {
                      refractRay.refract_index = mat->refract_index;
                    }
                    
                    globalCol = globalCol + mat->refract_cof * shadeRay(refractRay, depth+1, maxDepth);
                }
            }
        }
    }
        
    col = localCol + globalCol;

    col.clamp();
    return col;
}   

void Raytracer::render( int width, int height, Point3D eye, Vector3D view, 
        Vector3D up, double fov, char* fileName ) {
    Matrix4x4 viewToWorld;
    _scrWidth = width;
    _scrHeight = height;
    double factor = (double(height)/2)/tan(fov*M_PI/360.0);

    initPixelBuffer();
    viewToWorld = initInvViewMatrix(eye, view, up);
  
    // Additional paramters.    
    int max_depth = 4;
    int FSAA = 1;     // Display resolution multiplier for supersampling.

    // Construct a ray for each pixel.
    for (int i = 0; i < _scrHeight; i++) {
        std::cout << "Rendering row: " << i+1 << "/" << _scrHeight << std::endl;
        for (int j = 0; j < _scrWidth; j++) {

            // Sets up ray origin and direction in view space, 
            // image plane is at z = -1.
            Point3D origin(0, 0, 0);
            Point3D imagePlane;
            imagePlane[0] = (-double(width)/2 + 0.5 + j)/factor;
            imagePlane[1] = (-double(height)/2 + 0.5 + i)/factor;
            imagePlane[2] = -1;
                
            Colour col(0, 0, 0);
  
            // Subsampling of the pixel.
            for (int ss_i = 0; ss_i < FSAA; ss_i++) {
                for (int ss_j = 0; ss_j < FSAA; ss_j++) {
                    // Regular grid supersampling.
                    imagePlane[0] += (-0.5 + (ss_i + 0.5)/FSAA) / factor;
                    imagePlane[1] += (-0.5 + (ss_j + 0.5)/FSAA) / factor;

                    // Construct and transform ray to world co-ordinates.
                    Ray3D ray;
                    ray.origin = viewToWorld * imagePlane;
                    ray.dir = viewToWorld * (imagePlane - origin);
            
                    // Normalize ray direction for later shading/reflection
                    // computations.
                    ray.dir.normalize();

                    // Only intersections along the positive region of the
                    // ray are considered for shading.
                    ray.intersection.t_min = 0;
                    ray.intersection.t_max = std::numeric_limits<double>::infinity();

                    col = col + shadeRay(ray, 1, max_depth); 
                }
            }
    
            col = (1.0 / (FSAA * FSAA)) * col;
            col.clamp(); 

            _rbuffer[i*width+j] = int(col[0]*255);
            _gbuffer[i*width+j] = int(col[1]*255);
            _bbuffer[i*width+j] = int(col[2]*255);
        }
    }

    flushPixelBuffer(fileName);
}

int main(int argc, char* argv[])
{   
    // Build your scene and setup your camera here, by calling 
    // functions from Raytracer.  The code here sets up an example
    // scene and renders it from two different view points, DO NOT
    // change this if you're just implementing part one of the 
    // assignment.  
    Raytracer raytracer;
    int width = 320; 
    int height = 240; 
    
    if (argc == 3) {
        width = atoi(argv[1]);
        height = atoi(argv[2]);
    }
/*
    // SCENE 1 (DEFAULT)
    
    // Camera parameters.
    Point3D eye(0, 0, 1);
    Vector3D view(0, 0, -1);
    Vector3D up(0, 1, 0);
    double fov = 60;

    // Defines a material for shading.
    Material gold( Colour(0.3, 0.3, 0.3), Colour(0.75164, 0.60648, 0.22648), 
            Colour(0.628281, 0.555802, 0.366065), 
            51.2, 0.5, 0, 0 );
    Material jade( Colour(0, 0, 0), Colour(0.54, 0.89, 0.63), 
            Colour(0.316228, 0.316228, 0.316228), 
            12.8, 0.3, 0, 0 );

    // Defines a point light source.
    raytracer.addLightSource( new PointLight(Point3D(0, 0, 5), 
                Colour(0.9, 0.9, 0.9) ) );

    // Add a unit square into the scene with material mat.
    SceneDagNode* sphere = raytracer.addObject( new UnitSphere(), &gold );
    SceneDagNode* plane = raytracer.addObject( new UnitSquare(), &jade );
    
    // Apply some transformations to the unit square.
    double factor1[3] = { 1.0, 2.0, 1.0 };
    double factor2[3] = { 6.0, 6.0, 6.0 };
    raytracer.translate(sphere, Vector3D(0, 0, -5));    
    raytracer.rotate(sphere, 'x', -45); 
    raytracer.rotate(sphere, 'z', 45); 
    raytracer.scale(sphere, Point3D(0, 0, 0), factor1);

    raytracer.translate(plane, Vector3D(0, 0, -7)); 
    raytracer.rotate(plane, 'z', 45); 
    raytracer.scale(plane, Point3D(0, 0, 0), factor2);

    // Render the scene, feel free to make the image smaller for
    // testing purposes.    
    raytracer.render(width, height, eye, view, up, fov, "view1.bmp");
    
    // Render it from a different point of view.
    Point3D eye2(4, 2, 1);
    Vector3D view2(-4, -2, -6);
    raytracer.render(width, height, eye2, view2, up, fov, "view2.bmp");
    */


    // SCENE 2
    
    // Camera paramters
    Point3D eye(0,0,4);
    Vector3D view(0,0,-1);
    Vector3D up(0,1,0);
    double fov = 60;
    
    raytracer.set_glossy_reflections(true);
    
    // Materials :   ambient             |  diffuse             |     specular            |    exp |   reflectance | refractance |  index
    Material mat1(   Colour(0.1, 0.1, 0.1), Colour(0.5, 0.3, 0.2),    Colour(0.3, 0.3, 0.3),   200,    0.0,          0.5,           1.57);
    Material mat2(   Colour(0.1, 0.1, 0.1), Colour(0.0, 0.0, 0.0),    Colour(0.0, 0.0, 0.0),   100,    0.7,          0.0,            0.0);
    Material mat3(   Colour(0.0, 0.0, 0.0), Colour(0.6, 0.6, 0.6),    Colour(0.7, 0.7, 0.7),   100,    0.0,          0.0,            1.1);
    Material mat4(   Colour(0.1, 0.1, 0.1), Colour(0.55, 0.0, 0.0),   Colour(0.5, 0.5, 0.5),   200,    0.1,          0.0,            0.0);
    Material mat5(   Colour(0.2, 0.2, 0.2), Colour(0.09, 0.09, 0.44), Colour(0.4, 0.4, 0.4),   300,    0.0,          0.0,            0.0);
    Material mat6(   Colour(0.2, 0.2, 0.2), Colour(1.0, 0.76, 0.14),  Colour(0.4, 0.4, 0.4),   300,    0.0,          0.0,            0.0);
    Material walls3( Colour(0.1, 0.1, 0.1), Colour(0.62, 0.51, 0.32), Colour(0.0, 0.0, 0.0),     0,    0.0,          0.0,            0.0);
    Material walls2( Colour(0.1, 0.1, 0.1), Colour(0.62, 0.51, 0.32), Colour(0.0, 0.0, 0.0),     0,    0.1,          0.0,            0.0);
    Material walls(  Colour(0, 0.2, 0),     Colour(0.19, 0.20, 0.08), Colour(0.0, 0.0, 0.0),     0,    0.0,          0.0,            0.0);
    
    // Point light sources.
    //raytracer.addLightSource( new PointLight(Point3D(4,1,1), Colour(1.0, 1.0, 1.0)) );
    raytracer.addLightSource( new PointLight(Point3D(-8, 8.9, 4), Colour(1.0, 1.0, 1.0)) );
    raytracer.addLightSource( new PointLight(Point3D( 7, 6, 7),  Colour(0.5, 0.5, 0.5)) );
    //raytracer.addLightSource( 
    //    new AreaLight(Point3D(0.0, 5.0, 2.0), Vector3D(1.0, 0.0, 0.0), Vector3D(0.0, 0.0, 1.0), Colour(1.0, 1.0, 1.0)) );
    
    
    // Scene geometry.
    SceneDagNode* sphere1    = raytracer.addObject( new UnitSphere(), &mat3   );
    SceneDagNode* sphere2    = raytracer.addObject( new UnitSphere(), &mat2   );
    SceneDagNode* sphere3    = raytracer.addObject( new UnitSphere(), &mat2 );
    //SceneDagNode* cylinder1  = raytracer.addObject( new UnitCylinder(), &mat4 );
    //SceneDagNode* cylinder2  = raytracer.addObject( new UnitCylinder(), &mat5 );
    //SceneDagNode* cylinder3  = raytracer.addObject( new UnitCylinder(), &mat6 );
    SceneDagNode* floor      = raytracer.addObject( new UnitSquare(), &walls3 );
    SceneDagNode* backWall   = raytracer.addObject( new UnitSquare(), &walls3  );
    SceneDagNode* leftWall   = raytracer.addObject( new UnitSquare(), &walls3  );
    SceneDagNode* rightWall  = raytracer.addObject( new UnitSquare(), &walls3  );
    SceneDagNode* frontWall  = raytracer.addObject( new UnitSquare(), &walls3  );
    SceneDagNode* ceiling    = raytracer.addObject( new UnitSquare(), &walls3  );
    
    
    // Transformations.
    double factor1[3] = { 0.8, 0.8, 0.8 };
    double factor2[3] = {  20,  20,  20 };
    double factor3[3] = { 2, 2, 2};
    double factor4[3] = {1.5, 3, 1.5};
    double factor5[3] = {1.2, 2, 1.2};
    raytracer.translate(sphere1, Vector3D(-0.6, -0.7, 0.5));
    raytracer.scale(sphere1, Point3D(0,0,0), factor1);

    raytracer.translate(sphere2, Vector3D(2, 0.5, -1));
    raytracer.scale(sphere2, Point3D(0,0,0), factor3);
  
    raytracer.translate(sphere3, Vector3D(-3, 0.5, -4));
    raytracer.scale(sphere3, Point3D(0,0,0), factor3);

    //raytracer.translate(cylinder1, Vector3D(-2.5, -1.5, -1.7));
    //raytracer.scale(cylinder1, Point3D(0,0,0), factor4);

    //raytracer.translate(cylinder2, Vector3D(2, -1, 0.0));
    //raytracer.rotate(cylinder2, 'y', -45);
    //raytracer.rotate(cylinder2, 'x', 90);

    //raytracer.translate(cylinder3, Vector3D(0.6, -2, -0.7));
    //raytracer.scale(cylinder3, Point3D(0,0,0), factor5);
 


    raytracer.translate(floor, Vector3D(0, -1.5, 0));
    raytracer.rotate(floor, 'x', -89);
    raytracer.scale(floor, Point3D(0,0,0), factor2);

    raytracer.translate(backWall, Vector3D(0, 0, -10));
    raytracer.scale(backWall, Point3D(0,0,0), factor2);
    
    raytracer.translate(leftWall, Vector3D(-10, 0, 0));
    raytracer.rotate(leftWall, 'y', 90);
    raytracer.scale(leftWall, Point3D(0,0,0), factor2);
    
    raytracer.translate(rightWall, Vector3D(10, 0, 0));
    raytracer.rotate(rightWall, 'y', -90);
    raytracer.scale(rightWall, Point3D(0,0,0), factor2);
    
    raytracer.translate(frontWall, Vector3D(0, 0, 8));
    raytracer.rotate(frontWall, 'y', 180);
    raytracer.scale(frontWall, Point3D(0,0,0), factor2);
    
    raytracer.translate(ceiling, Vector3D(0, 9, 0));
    raytracer.rotate(ceiling, 'x', 89);
    raytracer.scale(ceiling, Point3D(0,0,0), factor2);


    // Render scene.
    raytracer.render(width, height, eye, view, up, fov, "render2.bmp");
    


    return 0;
}

