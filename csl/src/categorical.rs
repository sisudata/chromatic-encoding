//! An interface module which abstracts over different types of categorical
//! data featurization schemes.
//!
//! A covariate X (which, in this context, will always be categorical) spans over
//! a compact range of values 0..|X|, its covariate values.
//!
//! A feature represents a pair of a covariate and one of its attained values, e.g.,
//! X=3, but internally there is a single of features defined by indices 0..n.
//!
//! In `csl` we get access to data as a stream of feature lists for every training
//! instance as well as the target value, a label which is a real number.
//!
//! Coloring provides a mapping from features to colors. The colors are the covariates.
//! After we construct this mapping, we can assign each feature its covariate value.
//!
//! By construction, at most one value is attained per covariate in every training instance.
//! This gives an implicit dataframe which has as many columns as covariates, each with
//! some cardinal assignment (or null).
//!
//! This module provides a streaming featurization mechanism for this implicit dataframe,
//! using a provided budget in the number of output columns.

mod frequency_truncation;
mod identity;
pub(crate) mod sketch;
mod submodular_expansion;
mod target_encode;

use crate::feature::Featurizer;
use crate::svm_scanner::DelimIter;
use crate::svm_scanner::SvmScanner;
use std::io::Write;

#[derive(Debug, Clone, Copy)]
pub enum Compression {
    TargetEncode,
    SubmodularExpansion,
    SubmodularExpansionNoSplit,
    FrequencyTruncation,
    FieldAwareFrequencyTruncation,
    SubmodularSort,
    NoSplitSubmodularSort,
    Identity,
    Unbiased,
}

/// A learned categorical encoding dictionary
pub(crate) struct EncodingDictionary {
    variant: EncodingDictionaryVariant,
}

/// An encoder referring to a particular encoding dictionary encodes a line
/// of features into a line of encoded features.
///
/// The encoder is stateful. It starts out in a clean slate for the 'current example'.
/// Features can be added incrementally to the 'current example' using observe().
/// Then finish() dumps the current example, encoded, into the parameter file.
pub(crate) struct Encoder<'a> {
    dictionary: &'a EncodingDictionaryVariant,
    variant: EncoderVariant,
}

impl EncodingDictionary {
    pub(crate) fn new(
        compression: Compression,
        budget: usize,
        train: &SvmScanner,
        featurizer: &Featurizer,
        ncolors: usize,
        colors: Vec<u32>,
        split_rate: usize,
    ) -> Self {
        let variant = match compression {
            Compression::TargetEncode => EncodingDictionaryVariant::TargetEncode(
                target_encode::EncodingDictionary::new(budget, train, featurizer, ncolors, colors),
            ),
            Compression::SubmodularExpansion => EncodingDictionaryVariant::SubmodularExpansion(
                submodular_expansion::EncodingDictionary::new(
                    budget, train, featurizer, ncolors, colors, true, false, split_rate,
                ),
            ),
            Compression::SubmodularExpansionNoSplit => {
                EncodingDictionaryVariant::SubmodularExpansionNoSplit(
                    submodular_expansion::EncodingDictionary::new(
                        budget, train, featurizer, ncolors, colors, false, false, 0,
                    ),
                )
            }
            Compression::FrequencyTruncation => EncodingDictionaryVariant::FrequencyTruncation(
                frequency_truncation::EncodingDictionary::new(budget, featurizer),
            ),
            Compression::FieldAwareFrequencyTruncation => {
                EncodingDictionaryVariant::FieldAwareFrequencyTruncation(
                    frequency_truncation::FieldAwareDictionary::new(
                        budget, featurizer, ncolors, colors,
                    ),
                )
            }
            Compression::SubmodularSort => EncodingDictionaryVariant::SubmodularSort(
                submodular_expansion::EncodingDictionary::new(
                    budget, train, featurizer, ncolors, colors, true, true, split_rate,
                ),
            ),
            Compression::NoSplitSubmodularSort => EncodingDictionaryVariant::NoSplitSubmodularSort(
                submodular_expansion::EncodingDictionary::new(
                    budget, train, featurizer, ncolors, colors, false, true, 0,
                ),
            ),
            Compression::Identity => EncodingDictionaryVariant::Identity(
                identity::EncodingDictionary::new(budget, ncolors, colors),
            ),
            Compression::Unbiased => EncodingDictionaryVariant::Unbiased(
                identity::EncodingDictionary::new(budget, ncolors, colors),
            ),
        };
        Self { variant }
    }

    pub(crate) fn new_encoder<'a>(&'a self) -> Encoder<'a> {
        let variant = match self.variant {
            EncodingDictionaryVariant::TargetEncode(ref dictionary) => {
                EncoderVariant::TargetEncode(target_encode::Encoder::new(dictionary))
            }
            EncodingDictionaryVariant::SubmodularExpansion(ref dictionary) => {
                EncoderVariant::SubmodularExpansion(submodular_expansion::Encoder::new(dictionary))
            }
            EncodingDictionaryVariant::SubmodularExpansionNoSplit(ref dictionary) => {
                EncoderVariant::SubmodularExpansionNoSplit(submodular_expansion::Encoder::new(
                    dictionary,
                ))
            }
            EncodingDictionaryVariant::FrequencyTruncation(ref dictionary) => {
                EncoderVariant::FrequencyTruncation(frequency_truncation::Encoder::new(dictionary))
            }
            EncodingDictionaryVariant::FieldAwareFrequencyTruncation(ref dictionary) => {
                EncoderVariant::FieldAwareFrequencyTruncation(
                    frequency_truncation::FieldAwareEncoder::new(dictionary),
                )
            }
            EncodingDictionaryVariant::SubmodularSort(ref dictionary) => {
                EncoderVariant::SubmodularSort(submodular_expansion::Encoder::new(dictionary))
            }
            EncodingDictionaryVariant::NoSplitSubmodularSort(ref dictionary) => {
                EncoderVariant::NoSplitSubmodularSort(submodular_expansion::Encoder::new(
                    dictionary,
                ))
            }
            EncodingDictionaryVariant::Identity(ref dictionary) => {
                EncoderVariant::Identity(identity::Encoder::new(dictionary, false))
            }
            EncodingDictionaryVariant::Unbiased(ref dictionary) => {
                EncoderVariant::Unbiased(identity::Encoder::new(dictionary, true))
            }
        };
        Encoder {
            dictionary: &self.variant,
            variant,
        }
    }

    /// Returns the field dimensions up to the dense inputs. This is the number of features
    /// encoded in the cateogircal column, if any are present (columns with only 1 feature
    /// must be 1-hot or numeric).
    pub(crate) fn field_dims(&self) -> Vec<u32> {
        match self.variant {
            EncodingDictionaryVariant::FieldAwareFrequencyTruncation(ref dictionary) => {
                dictionary.field_dims()
            }
            EncodingDictionaryVariant::Identity(ref dictionary) => dictionary.field_dims(),
            _ => {
                let encoder = self.new_encoder();
                vec![1; encoder.dense_offset()]
            }
        }
    }
}

impl<'a> Encoder<'a> {
    // POTENTIAL ALGORITHMIC IMPROVEMENT: for leave-one-out style encoders we'll need
    // an optional observe_target() which should be called before any observe() during
    // training.
    pub(crate) fn observe(&mut self, feature: u32) {
        match (self.dictionary, &mut self.variant) {
            (
                EncodingDictionaryVariant::TargetEncode(dictionary),
                EncoderVariant::TargetEncode(ref mut variant),
            ) => variant.observe(dictionary, feature),
            (
                EncodingDictionaryVariant::SubmodularExpansion(dictionary),
                EncoderVariant::SubmodularExpansion(ref mut variant),
            ) => variant.observe(dictionary, feature),
            (
                EncodingDictionaryVariant::SubmodularExpansionNoSplit(dictionary),
                EncoderVariant::SubmodularExpansionNoSplit(ref mut variant),
            ) => variant.observe(dictionary, feature),
            (
                EncodingDictionaryVariant::FrequencyTruncation(dictionary),
                EncoderVariant::FrequencyTruncation(ref mut variant),
            ) => variant.observe(dictionary, feature),
            (
                EncodingDictionaryVariant::FieldAwareFrequencyTruncation(dictionary),
                EncoderVariant::FieldAwareFrequencyTruncation(ref mut variant),
            ) => variant.observe(dictionary, feature),
            (
                EncodingDictionaryVariant::SubmodularSort(dictionary),
                EncoderVariant::SubmodularSort(ref mut variant),
            ) => variant.observe(dictionary, feature),
            (
                EncodingDictionaryVariant::NoSplitSubmodularSort(dictionary),
                EncoderVariant::NoSplitSubmodularSort(ref mut variant),
            ) => variant.observe(dictionary, feature),
            (
                EncodingDictionaryVariant::Identity(dictionary),
                EncoderVariant::Identity(ref mut variant),
            ) => variant.observe(dictionary, feature),
            (
                EncodingDictionaryVariant::Unbiased(dictionary),
                EncoderVariant::Unbiased(ref mut variant),
            ) => variant.observe(dictionary, feature),
            _ => panic!("mismatched dictionary/encoder variants"),
        }
    }

    /// The offset at which dense features should be written.
    ///
    /// This will typically end up being the budget, and it is the number
    /// of feature columns that are sparse (so that the dense columns come after).
    pub(crate) fn dense_offset(&self) -> usize {
        match &self.variant {
            EncoderVariant::TargetEncode(ref variant) => variant.dense_offset(),
            EncoderVariant::SubmodularExpansion(ref variant) => variant.dense_offset(),
            EncoderVariant::SubmodularExpansionNoSplit(ref variant) => variant.dense_offset(),
            EncoderVariant::FrequencyTruncation(ref variant) => variant.dense_offset(),
            EncoderVariant::FieldAwareFrequencyTruncation(ref variant) => variant.dense_offset(),
            EncoderVariant::SubmodularSort(ref variant) => variant.dense_offset(),
            EncoderVariant::NoSplitSubmodularSort(ref variant) => variant.dense_offset(),
            EncoderVariant::Identity(ref variant) => variant.dense_offset(),
            EncoderVariant::Unbiased(ref variant) => variant.dense_offset(),
        }
    }

    /// Some encoders rely on the dictionary being initialized from a disjoint
    /// subset of the data than what will be used for training, so this method
    /// can be used to skip examples when writing examples back out.
    pub(crate) fn skip_example(&self, d: DelimIter<'_>) -> bool {
        match &self.variant {
            EncoderVariant::TargetEncode(_) => false,
            EncoderVariant::SubmodularExpansion(e) => e.skip_example(d),
            EncoderVariant::SubmodularExpansionNoSplit(_) => false,
            EncoderVariant::FrequencyTruncation(_) => false,
            EncoderVariant::FieldAwareFrequencyTruncation(_) => false,
            EncoderVariant::SubmodularSort(e) => e.skip_example(d),
            EncoderVariant::NoSplitSubmodularSort(_) => false,
            EncoderVariant::Identity(_) => false,
            EncoderVariant::Unbiased(_) => false,
        }
    }

    pub(crate) fn finish<W: Write>(&mut self, writer: &mut W) {
        match &mut self.variant {
            EncoderVariant::TargetEncode(ref mut variant) => variant.finish(writer),
            EncoderVariant::SubmodularExpansion(ref mut variant) => variant.finish(writer),
            EncoderVariant::SubmodularExpansionNoSplit(ref mut variant) => variant.finish(writer),
            EncoderVariant::FrequencyTruncation(ref mut variant) => variant.finish(writer),
            EncoderVariant::FieldAwareFrequencyTruncation(ref mut variant) => {
                variant.finish(writer)
            }
            EncoderVariant::SubmodularSort(ref mut variant) => variant.finish(writer),
            EncoderVariant::NoSplitSubmodularSort(ref mut variant) => variant.finish(writer),
            EncoderVariant::Identity(ref mut variant) => variant.finish(writer),
            EncoderVariant::Unbiased(ref mut variant) => variant.finish(writer),
        }
    }
}

enum EncodingDictionaryVariant {
    TargetEncode(target_encode::EncodingDictionary),
    SubmodularExpansion(submodular_expansion::EncodingDictionary),
    SubmodularExpansionNoSplit(submodular_expansion::EncodingDictionary),
    FrequencyTruncation(frequency_truncation::EncodingDictionary),
    FieldAwareFrequencyTruncation(frequency_truncation::FieldAwareDictionary),
    SubmodularSort(submodular_expansion::EncodingDictionary),
    NoSplitSubmodularSort(submodular_expansion::EncodingDictionary),
    Identity(identity::EncodingDictionary),
    Unbiased(identity::EncodingDictionary),
}

enum EncoderVariant {
    TargetEncode(target_encode::Encoder),
    SubmodularExpansion(submodular_expansion::Encoder),
    SubmodularExpansionNoSplit(submodular_expansion::Encoder),
    FrequencyTruncation(frequency_truncation::Encoder),
    FieldAwareFrequencyTruncation(frequency_truncation::FieldAwareEncoder),
    SubmodularSort(submodular_expansion::Encoder),
    NoSplitSubmodularSort(submodular_expansion::Encoder),
    Identity(identity::Encoder),
    Unbiased(identity::Encoder),
}
