//! The Sinsemilla hash function and commitment scheme.

#![no_std]

use core::iter;

#[cfg(not(feature = "sinsemilla_table"))]
use group::Curve;
//use group::Wnaf;
use pasta_curves::arithmetic::{CurveAffine, CurveExt};
use pasta_curves::pallas;
use subtle::{Choice, CtOption};

//use crate::spec::{extract_p_bottom, i2lebsp};

#[cfg(feature = "precomputed-tables")]
mod constants;

#[cfg(feature = "precomputed-tables")]
use constant::SINSEMILLA_S;

pub const K: usize = 10;
pub const Q_PERSONALIZATION: &str = "z.cash:SinsemillaQ";
pub const S_PERSONALIZATION: &str = "z.cash:SinsemillaS";

fn extract_p(point: &pallas::Point) -> pallas::Base {
    point
        .to_affine()
        .coordinates()
        .map(|c| *c.x())
        .unwrap_or_else(pallas::Base::zero)
}

struct Pad<I: Iterator<Item = bool>>(iter::Peekable<I>, bool);

impl<I: Iterator<Item = bool>> Pad<I> {
    fn new(inner: I) -> Self {
        Pad(inner.peekable(), false)
    }
}

impl<I: Iterator<Item = bool>> Iterator for Pad<I> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0.peek().is_none() {
            None
        } else {
            Some(
                (0..K)
                    .into_iter()
                    .map(|_| self.0.next().unwrap_or(false))
                    .enumerate()
                    .fold(0u32, |acc, (i, b)| acc + if b { 1 << i } else { 0 }),
            )
        }
    }
}

#[derive(Debug, Clone)]
#[allow(non_snake_case)]
pub struct HashDomain {
    Q: pallas::Point,
}

impl HashDomain {
    pub fn new(domain: &str) -> Self {
        HashDomain {
            Q: pallas::Point::hash_to_curve(Q_PERSONALIZATION)(domain.as_bytes()),
        }
    }

    #[allow(non_snake_case)]
    pub fn hash_to_point(&self, msg: impl Iterator<Item = bool>) -> pallas::Point {
        #[cfg(not(feature = "sinsemilla_table"))]
        let hasher = pallas::Point::hash_to_curve(S_PERSONALIZATION);

        Pad::new(msg).fold(self.Q, |acc, n| {
            #[cfg(feature = "sinsemilla_table")]
            {
                let (S_x, S_y) = SINSEMILLA_S[n as usize];
                let S_chunk = pallas::Affine::from_xy(S_x, S_y).unwrap();
                (acc + S_chunk) + acc
            }
            #[cfg(not(feature = "sinsemilla_table"))]
            {
                let S_chunk = hasher(&n.to_le_bytes()).to_affine();
                (acc + S_chunk) + acc
            }
        })
    }

    pub fn hash(&self, msg: impl Iterator<Item = bool>) -> CtOption<pallas::Base> {
        CtOption::new(extract_p(&self.hash_to_point(msg)), Choice::from(1))
    }

    /// Returns the Sinsemilla $Q$ constant for this domain.
    #[cfg(test)]
    #[allow(non_snake_case)]
    pub(crate) fn Q(&self) -> pallas::Point {
        self.Q
    }
}

#[derive(Debug)]
#[allow(non_snake_case)]
pub struct CommitDomain {
    M: HashDomain,
    R: pallas::Point,
}

impl CommitDomain {
    pub fn new(domain: &str) -> Self {
        //let m_prefix = format!("{}-M", domain);
        //let r_prefix = format!("{}-r", domain);

        //let note: &str = "z.cash:Orchard-NoteCommit";
        //let ivk: &str = "z.cash:Orchard-CommitIvk";

        let (m_prefix, r_prefix) = match domain {
            "z.cash:Orchard-NoteCommit" => {
                ("z.cash:Orchard-NoteCommit-M", "z.cash:Orchard-NoteCommit-r")
            }
            "z.cash:Orchard-CommitIvk" => {
                ("z.cash:Orchard-CommitIvk-M", "z.cash:Orchard-CommitIvk-r")
            }
            _ => panic!("unexpected domain"),
        };

        let hasher_r = pallas::Point::hash_to_curve(&r_prefix);
        CommitDomain {
            M: HashDomain::new(&m_prefix),
            R: hasher_r(&[]),
        }
    }

    #[allow(non_snake_case)]
    pub fn commit(
        &self,
        msg: impl Iterator<Item = bool>,
        r: &pallas::Scalar,
    ) -> CtOption<pallas::Point> {
        // We use complete addition for the blinding factor.
        CtOption::new(self.M.hash_to_point(msg) + self.R * r, Choice::from(1))
        //.map(|p| p + Wnaf::new().scalar(r).base(self.R)) // TODO
    }

    pub fn short_commit(
        &self,
        msg: impl Iterator<Item = bool>,
        r: &pallas::Scalar,
    ) -> CtOption<pallas::Base> {
        CtOption::new(extract_p(&self.commit(msg, r).unwrap()), Choice::from(1))
    }

    #[cfg(test)]
    #[allow(non_snake_case)]
    pub(crate) fn R(&self) -> pallas::Point {
        self.R
    }
}
