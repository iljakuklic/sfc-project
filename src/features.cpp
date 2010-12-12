/*
 * SFC project (2010) - music genre classifier
 * by Lukas Kuklinek <xkukli01@stud.fit.vutbr.cz>
 * Faculty of Information Tachnology
 * Brno University of Technology
 */


#include "features.hpp"
#include <stdexcept>
#include <fstream>
#include <WaveFile.h>
#include <feature/MfccExtractor.h>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/filesystem.hpp>

const unsigned FRAME_LENGTH = 30;
const unsigned PARAMS_PER_FRAME = 15;
const double FRAME_OVERLAP = 0.66;
const double PREEMPHASIS_FACTOR = 0.9375;

Features::Features(const std::string& filename) :
    fea(new Aquila::MfccExtractor(FRAME_LENGTH, PARAMS_PER_FRAME))
{
    Aquila::WaveFile wav(FRAME_LENGTH, FRAME_OVERLAP);
    wav.load(filename);
    if (wav.getFramesCount() < 2)
        throw std::runtime_error("record is not long enough");
    Aquila::TransformOptions options;
    options.preemphasisFactor = PREEMPHASIS_FACTOR;
    options.windowType = Aquila::WIN_HAMMING;
    options.zeroPaddedLength = wav.getSamplesPerFrameZP();
    fea->process(&wav, options);
}

Features::~Features() { }

size_t Features::frames() const
{
    return raw_frames() - 1;
}

FeatureVector Features::feature(size_t frame) const
{
    FeatureVector v = raw_feature(frame);
    size_t s = v.size();
    v.resize(2 * s);

    // add first differential
    for (size_t i = 0; i < s; ++i)
        v[s + i] = raw_feature(frame + 1)[i] - v[i];

    return v;
}

size_t Features::raw_frames() const
{
    return fea->getFramesCount();
}

FeatureVector Features::raw_feature(size_t frame) const
{
    const std::vector<Real>& vec = fea->getVector(frame);
    FeatureVector ret(vec.size());
    std::copy(vec.begin(), vec.end(), ret.begin());
    return ret;
}


AnnotationType DataSet::load_annotations(const std::string& filename)
{
    AnnotationType result;
    std::string tag;
    Real score;
    std::ifstream ifs(filename.c_str());
    if (!ifs) throw std::runtime_error("Unable to open annotations file " + filename);

    while (ifs >> score) {
        getline(ifs, tag);
        result[tag.substr(tag.find_first_not_of(' '))] = score;
        //std::cout << tag.substr(tag.find_first_not_of(' ')) << ": " << score << std::endl;
    }

    return result;
}



using boost::numeric::ublas::scalar_vector;


size_t DataSet::count() const
{
    return samples.size();
}

FeatureVector DataSet::mean() const
{
    return element_div(sum, scalar_vector<Real>(sum.size(), count()));
}

FeatureVector DataSet::stddev() const
{
    FeatureVector m = mean();
    FeatureVector v = sumsq;
    for (size_t i = 0; i < v.size(); ++i) v(i) = v(i) / count() - m(i) * m(i);
    return v;
}

DataSet::DataSet() : /*labels(l),*/ normalized(false) {}

void DataSet::load(const std::string& filename_base, const LabelList& labels)
{
    FeatureVector out;

    if (labels.size() != 0) {
        AnnotationType a = load_annotations(filename_base + ".tag");
        out.resize(labels.size());
        // clamp every desired input and take logarithm
        for (size_t i = 0; i < labels.size(); ++i)
            out[i] = std::log(std::max(0.002, std::min(0.998, a[labels[i]] / 100.0)));
    }

    // load MFCC coefficients and associate them with desired output
    Features f(filename_base + ".wav");
    for (size_t i = 0; i < f.frames(); ++i)
        add_sample(f.feature(i), out);
}

void DataSet::load_dir(const std::string& dirname, const LabelList& labels)
{
    using boost::filesystem::directory_iterator;
    using boost::filesystem::path;

    path dirpath(dirname);
    if (!exists(dirpath)) throw std::runtime_error("Directory '" + dirname + "' does not exist.");
    directory_iterator dirend;
    
    for (directory_iterator dir(dirpath); dir != dirend; ++dir)
        if (dir->path().extension() == ".wav")
            load((dir->path().parent_path() / dir->path().stem()).string(), labels);
}

void DataSet::load_tmp(const std::string& filename)
{
    std::ifstream ifs(filename.c_str());
    FeatureVector in, out;
    ifs >> sum >> sumsq >> normalized;
    while (ifs) {
        ifs >> in >> out;
        samples.push_back(std::make_pair(in, out));
    }
}

void DataSet::write_tmp(const std::string& filename) const
{
    if (samples.size() == 0) throw std::runtime_error("Nothing to output.");

    std::ofstream ofs(filename.c_str());
    ofs << sum << ' ' << sumsq << ' ' << normalized << std::endl;
    for (size_t i = 0; i < samples.size(); ++i)
        ofs << samples[i].first << ' ' << samples[i].second << std::endl;
}

void DataSet::shuffle() { std::random_shuffle(samples.begin(), samples.end()); }

void DataSet::clear() { samples.clear(); normalized = false; sum = sumsq = FeatureVector(); }

void DataSet::normalize(FeatureVector& vec) const { vec = element_div(vec - mean(), stddev()); }

void DataSet::add_sample(const FeatureVector& in, const FeatureVector& out)
{
    if (normalized) throw std::logic_error("Adding data to a normalized dataset.");

    if (sum.size() == 0)
        sum = sumsq = boost::numeric::ublas::zero_vector<Real>(in.size());

    sum   += in;
    sumsq += element_prod(in, in); // in^2
    samples.push_back(std::make_pair(in, out));
}

void DataSet::normalize_all(FeatureVector m, FeatureVector s)
{
    if (normalized || samples.size() <= 1) return;

    if (m.size() == 0) {
        m = mean();
        s = stddev();
    }

    for (size_t i = 0; i < samples.size(); ++i)
        samples[i].first = element_div(samples[i].first - m, s);

    normalized = true;
}

//const LabelList& DataSet::get_labels() const { return labels; }

