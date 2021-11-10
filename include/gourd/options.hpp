
/* Enums for various global program options
 */


#ifndef _GOURD_OPTIONS_
#define _GOURD_OPTIONS_


namespace gourd {

  
  /* ****************************************************************/
  /*! Covariance function options
   */
  enum class cov_code {
    rbf,    /*!< Radial basis covariance (3 parameters) */
    rq,     /*!< Rational quadratic covariance (3 parameters) */
    matern  /*!< Matern covariance (3 parameters) */
  };

  
  /* ****************************************************************/
  /*! Available distance metrics
   */
  enum class dist_code {
    euclidean,    /*!< Euclidean distance */
    great_circle  /*!< Great-circle distance for spherical surfaces */
  };

  
};
// namespace gourd


#endif  // _GOURD_OPTIONS_
