/***********************************************************
     Starter code for Assignment 3

     This code was originally written by Jack Wang for
		    CSC418, SPRING 2005

		   light source classes

***********************************************************/

#include "util.h"

// Base class for a light source.  You could define different types
// of lights here, but point light is sufficient for most scenes you
// might want to render.  Different light sources shade the ray 
// differently.
class LightSource {

  public:

    // TODO: COMMENT ME.
    enum LIGHT_TYPE { POINT_LIGHT, AREA_LIGHT };
        

	  virtual Colour shade( Ray3D& ) = 0;
    virtual Colour ambientShade( Ray3D& ) = 0;

	  virtual Point3D get_position() const = 0; 
    virtual LIGHT_TYPE get_type() const = 0;
};


// A point light is defined by its position in world space and its
// colour.
class PointLight : public LightSource {

  public:
	  PointLight( Point3D pos, Colour col ) : _pos(pos), _col_ambient(col), 
	              _col_diffuse(col), _col_specular(col) {}

	  PointLight( Point3D pos, Colour ambient, Colour diffuse, Colour specular ) 
	            : _pos(pos), _col_ambient(ambient), _col_diffuse(diffuse), 
	              _col_specular(specular) {}

	  Colour shade( Ray3D& ray );
    Colour ambientShade( Ray3D& ray );

	  Point3D get_position() const { return _pos; }
    LIGHT_TYPE get_type() const { return POINT_LIGHT; }
	
  private:
  
	  Point3D _pos;
	  Colour _col_ambient;
	  Colour _col_diffuse; 
	  Colour _col_specular; 
};


// TODO: COMMET ME.
class AreaLight : public LightSource {

  public:
    
	  AreaLight( Point3D corner, Vector3D axis1, Vector3D axis2, Colour col ) 
            : _pos(corner + 0.5*axis1 + 0.5*axis2),
              _corner(corner), 
              _axis1(axis1), _axis2(axis2),
              _col_ambient(col), _col_diffuse(col), _col_specular(col) {}

	  AreaLight( Point3D corner, Vector3D axis1, Vector3D axis2, Colour ambient, Colour diffuse, Colour specular ) 
          : _pos(corner + 0.5*axis1 + 0.5*axis2),
            _corner(corner),
            _axis1(axis1), _axis2(axis2),
            _col_ambient(ambient), _col_diffuse(diffuse), _col_specular(specular) {}

    Colour shade( Ray3D& );
    Colour ambientShade( Ray3D& );

    Point3D get_position() const;
    LIGHT_TYPE get_type() const { return AREA_LIGHT; }
	
  private:
    
    Point3D _pos;
    Point3D _corner;
    Vector3D _axis1;
    Vector3D _axis2;
    
    Colour _col_ambient;
    Colour _col_diffuse;
    Colour _col_specular;
};
