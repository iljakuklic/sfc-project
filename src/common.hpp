/*
 * SFC project (2010) - music genre classifier
 * by Lukas Kuklinek <xkukli01@stud.fit.vutbr.cz>
 * Faculty of Information Tachnology
 * Brno University of Technology
 */


#pragma once
#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <string>
#include <vector>
#include <cmath>
#include <boost/numeric/ublas/vector.hpp>

/// floating-point type for feature representation
typedef double Real;

/// feature vector for single window
typedef boost::numeric::ublas::vector<Real> FeatureVector;

/// label list type
typedef std::vector<std::string> LabelList;

/// sigmoid function
inline Real sigmoid(Real x) { return 1.0 / (1.0 + std::exp(-x)); }
/// sigmoid function logarithm
inline Real logsigmoid(Real x) { return -std::log(1.0 + std::exp(-x)); }

#endif // COMMON_HPP_

