//! PKCS#8 documents: serialized PKCS#8 private keys and SPKI public keys
// TODO(tarcieri): heapless support?

use crate::{Error, PrivateKeyInfo, Result, SubjectPublicKeyInfo};
use alloc::{borrow::ToOwned, vec::Vec};
use core::{
    convert::{TryFrom, TryInto},
    fmt,
};
use zeroize::{Zeroize, Zeroizing};

#[cfg(feature = "std")]
use std::{fs, path::Path, str};

#[cfg(feature = "pem")]
use {crate::pem, alloc::string::String, core::str::FromStr};

/// PKCS#8 private key document
///
/// This type provides storage for a PKCS#8 private key encoded as ASN.1 DER
/// with the invariant that the contained-document is "well-formed", i.e. it
/// will parse successfully according to this crate's parsing rules.
#[derive(Clone)]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub struct PrivateKeyDocument(Zeroizing<Vec<u8>>);

impl PrivateKeyDocument {
    /// Parse [`PrivateKeyDocument`] from ASN.1 DER-encoded PKCS#8
    pub fn from_der(bytes: &[u8]) -> Result<Self> {
        bytes.try_into()
    }

    /// Parse [`PrivateKeyDocument`] from PEM-encoded PKCS#8.
    ///
    /// PEM-encoded private keys can be identified by the leading delimiter:
    ///
    /// ```text
    /// -----BEGIN PRIVATE KEY-----
    /// ```
    #[cfg(feature = "pem")]
    #[cfg_attr(docsrs, doc(cfg(feature = "pem")))]
    pub fn from_pem(s: &str) -> Result<Self> {
        let der_bytes = pem::decode(s, pem::PRIVATE_KEY_BOUNDARY)?;
        Self::from_der(&*der_bytes)
    }

    /// Serialize [`PrivateKeyDocument`] as self-zeroizing PEM-encoded PKCS#8 string.
    #[cfg(feature = "pem")]
    #[cfg_attr(docsrs, doc(cfg(feature = "pem")))]
    pub fn to_pem(&self) -> Zeroizing<String> {
        Zeroizing::new(pem::encode(&self.0, pem::PRIVATE_KEY_BOUNDARY))
    }

    /// Load [`PrivateKeyDocument`] from an ASN.1 DER-encoded file on the local
    /// filesystem (binary format).
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    pub fn read_der_file(path: impl AsRef<Path>) -> Result<Self> {
        fs::read(path)?.try_into()
    }

    /// Load [`PrivateKeyDocument`] from a PEM-encoded file on the local filesystem.
    #[cfg(all(feature = "pem", feature = "std"))]
    #[cfg_attr(docsrs, doc(cfg(feature = "pem")))]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    pub fn read_pem_file(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = Zeroizing::new(fs::read(path)?);
        let pem = str::from_utf8(&*bytes).map_err(|_| Error::Decode)?;
        Self::from_pem(pem)
    }

    /// Write ASN.1 DER-encoded PKCS#8 private key to the given path
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    pub fn write_der_file(&self, path: impl AsRef<Path>) -> Result<()> {
        write_secret_file(path, self.as_ref())
    }

    /// Write PEM-encoded PKCS#8 private key to the given path
    #[cfg(all(feature = "pem", feature = "std"))]
    #[cfg_attr(docsrs, doc(cfg(feature = "pem")))]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    pub fn write_pem_file(&self, path: impl AsRef<Path>) -> Result<()> {
        write_secret_file(path, self.to_pem().as_bytes())
    }

    /// Parse the [`PrivateKeyInfo`] contained in this [`PrivateKeyDocument`]
    pub fn private_key_info(&self) -> PrivateKeyInfo<'_> {
        PrivateKeyInfo::from_der(self.0.as_ref()).expect("constructor failed to validate document")
    }
}

impl AsRef<[u8]> for PrivateKeyDocument {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

impl TryFrom<&[u8]> for PrivateKeyDocument {
    type Error = Error;

    fn try_from(bytes: &[u8]) -> Result<Self> {
        // Ensure document is well-formed
        PrivateKeyInfo::from_der(bytes)?;
        Ok(Self(Zeroizing::new(bytes.to_owned())))
    }
}

impl TryFrom<Vec<u8>> for PrivateKeyDocument {
    type Error = Error;

    fn try_from(mut bytes: Vec<u8>) -> Result<Self> {
        // Ensure document is well-formed
        if PrivateKeyInfo::from_der(&bytes).is_ok() {
            Ok(Self(Zeroizing::new(bytes)))
        } else {
            bytes.zeroize();
            Err(Error::Decode)
        }
    }
}

impl fmt::Debug for PrivateKeyDocument {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_tuple("PrivateKeyDocument")
            .field(&self.private_key_info())
            .finish()
    }
}

#[cfg(feature = "pem")]
#[cfg_attr(docsrs, doc(cfg(feature = "pem")))]
impl FromStr for PrivateKeyDocument {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::from_pem(s)
    }
}

/// SPKI public key document
///
/// This type provides storage for a SPKI public key encoded as ASN.1 DER with
/// the invariant that the contained-document is "well-formed", i.e. it will
/// parse successfully according to this crate's parsing rules.
#[derive(Clone)]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub struct PublicKeyDocument(Vec<u8>);

impl PublicKeyDocument {
    /// Parse [`PublicKeyDocument`] from ASN.1 DER
    pub fn from_der(bytes: &[u8]) -> Result<Self> {
        bytes.try_into()
    }

    /// Parse [`PublicKeyDocument`] from PEM
    ///
    /// PEM-encoded public keys can be identified by the leading delimiter:
    ///
    /// ```text
    /// -----BEGIN PUBLIC KEY-----
    /// ```
    #[cfg(feature = "pem")]
    #[cfg_attr(docsrs, doc(cfg(feature = "pem")))]
    pub fn from_pem(s: &str) -> Result<Self> {
        let der_bytes = pem::decode(s, pem::PUBLIC_KEY_BOUNDARY)?;
        Self::from_der(&*der_bytes)
    }

    /// Serialize [`PublicKeyDocument`] as PEM-encoded PKCS#8 string.
    #[cfg(feature = "pem")]
    #[cfg_attr(docsrs, doc(cfg(feature = "pem")))]
    pub fn to_pem(&self) -> String {
        pem::encode(&self.0, pem::PUBLIC_KEY_BOUNDARY)
    }

    /// Load [`PublicKeyDocument`] from an ASN.1 DER-encoded file on the local
    /// filesystem (binary format).
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    pub fn read_der_file(path: impl AsRef<Path>) -> Result<Self> {
        fs::read(path)?.try_into()
    }

    /// Load [`PublicKeyDocument`] from a PEM-encoded file on the local filesystem.
    #[cfg(all(feature = "pem", feature = "std"))]
    #[cfg_attr(docsrs, doc(cfg(feature = "pem")))]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    pub fn read_pem_file(path: impl AsRef<Path>) -> Result<Self> {
        let pem = fs::read_to_string(path)?;
        Self::from_pem(&pem)
    }

    /// Write ASN.1 DER-encoded public key to the given path
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    pub fn write_der_file(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, self.as_ref())?;
        Ok(())
    }

    /// Write PEM-encoded public key to the given path
    #[cfg(all(feature = "pem", feature = "std"))]
    #[cfg_attr(docsrs, doc(cfg(feature = "pem")))]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    pub fn write_pem_file(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, self.to_pem().as_bytes())?;
        Ok(())
    }

    /// Parse the [`SubjectPublicKeyInfo`] contained in this [`PublicKeyDocument`]
    pub fn spki(&self) -> SubjectPublicKeyInfo<'_> {
        SubjectPublicKeyInfo::from_der(self.0.as_ref())
            .expect("constructor failed to validate document")
    }
}

impl AsRef<[u8]> for PublicKeyDocument {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

impl TryFrom<&[u8]> for PublicKeyDocument {
    type Error = Error;

    fn try_from(bytes: &[u8]) -> Result<Self> {
        // Ensure document is well-formed
        SubjectPublicKeyInfo::from_der(bytes)?;
        Ok(Self(bytes.to_owned()))
    }
}

impl TryFrom<Vec<u8>> for PublicKeyDocument {
    type Error = Error;

    fn try_from(bytes: Vec<u8>) -> Result<Self> {
        // Ensure document is well-formed
        SubjectPublicKeyInfo::from_der(&bytes)?;
        Ok(Self(bytes))
    }
}

impl fmt::Debug for PublicKeyDocument {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_tuple("PublicKeyDocument")
            .field(&self.spki())
            .finish()
    }
}

#[cfg(feature = "pem")]
#[cfg_attr(docsrs, doc(cfg(feature = "pem")))]
impl FromStr for PublicKeyDocument {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::from_pem(s)
    }
}

/// Write a file containing secret data to the filesystem, restricting the
/// file permissions so it's only readable by the owner
#[cfg(all(unix, feature = "std"))]
fn write_secret_file(path: impl AsRef<Path>, data: &[u8]) -> Result<()> {
    use std::{io::Write, os::unix::fs::OpenOptionsExt};

    /// File permissions for secret data
    #[cfg(unix)]
    const SECRET_FILE_PERMS: u32 = 0o600;

    fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .mode(SECRET_FILE_PERMS)
        .open(path)
        .and_then(|mut file| file.write_all(data))?;

    Ok(())
}

/// Write a file containing secret data to the filesystem
// TODO(tarcieri): permissions hardening on Windows
#[cfg(all(not(unix), feature = "std"))]
fn write_secret_file(path: impl AsRef<Path>, data: &[u8]) -> Result<()> {
    fs::write(path, data)?;
    Ok(())
}
