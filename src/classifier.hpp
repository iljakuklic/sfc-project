/*
 * SFC project (2010) - music genre classifier
 * by Lukas Kuklinek <xkukli01@stud.fit.vutbr.cz>
 * Faculty of Information Tachnology
 * Brno University of Technology
 */

#pragma once
#ifndef CLASSIFIER_HPP_
#define CLASSIFIER_HPP_

#include "neuralnet.hpp"
#include "features.hpp"

/**
 * Classifier capable of merging several results together.
 * One sound record usually yields several feature vectors.
 * This classifier blends them together into a single result.
 */
class Classifier {
    public:
        typedef NNLayer::Numeric Numeric;
        typedef NNLayer::Vector Vector;

    public:

        /// default constructor
        explicit Classifier();
        
        /// classify single data frame
        Vector exec(const Vector& in) const;
        /// classify dataset
        Vector exec(const DataSet& data) const;

        /// load from input stream
        void load(std::istream& is);

        /// get mean
        const Vector& mean() const;
        /// get standard deviation
        const Vector& stddev() const;
        /// get neural network
        const NeuralNet& neural_net() const;
        /// get output labels
        const LabelList& labels() const;

    private:
        NeuralNet nn;          ///< neural network
        Vector mean_, stddev_; ///< normalization constants
        LabelList labels_;     ///< output labels
};

// IO
std::istream& operator>>(std::istream& is, Classifier& c);
std::ostream& operator<<(std::ostream& os, const Classifier& c);

#endif // CLASSIFIER_HPP_

