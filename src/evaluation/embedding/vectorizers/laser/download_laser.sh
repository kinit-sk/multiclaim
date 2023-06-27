#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
# 
#-------------------------------------------------------
#
# This bash script installs NLLB LASER2 and LASER3 sentence encoders from Amazon s3

# default to download to current directory
mdir="cache/laser"

echo "Directory for model download: ${mdir}"

version=1  # model version

echo "Downloading networks..."

if [ ! -d ${mdir} ] ; then
  echo " - creating model directory: ${mdir}"
  mkdir -p ${mdir}
fi

function download {
    file=$1
    if [ -f ${mdir}/${file} ] ; then
        echo " - ${mdir}/$file already downloaded";
    else
        echo " - $s3/${file}";
        wget -q $s3/${file};
    fi   
}

MKDIR () {
  dname=$1
  if [ ! -d ${dname} ] ; then
    echo " - creating directory ${dname}"
    mkdir -p ${dname}
  fi
}

tools_ext="tools-external"

InstallMosesTools () {
  moses_git="https://raw.githubusercontent.com/moses-smt/mosesdecoder/RELEASE-4.0/scripts"
  moses_files=("tokenizer/normalize-punctuation.perl" \
               "tokenizer/remove-non-printing-char.perl" \ 
               "tokenizer/deescape-special-chars.perl" \ 
               "tokenizer/basic-protected-patterns" \
              )

  wdir="${tools_ext}/moses-tokenizer"
  MKDIR ${wdir}
  cd ${wdir}

  for f in ${moses_files[@]} ; do
    if [ ! -f `basename ${f}` ] ; then
      echo " - download ${f}"
      wget -q ${moses_git}/${f}
    fi
  done
  chmod 755 *perl
}


cd ${mdir}  # move to model directory

# available encoders
s3="https://dl.fbaipublicfiles.com/nllb/laser"

# LASER2 (download by default)
if [ ! -f ${mdir}/laser2.pt ] ; then
    echo " - $s3/laser2.pt"
    wget --trust-server-names -q https://tinyurl.com/nllblaser2
else 
    echo " - ${mdir}/laser2.pt already downloaded"
fi
download "laser2.spm"
download "laser2.cvocab"

echo "Installing external tools"
InstallMosesTools