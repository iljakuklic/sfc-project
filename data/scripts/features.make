#!/usr/bin/make -f
# feature extraction makefile

OUT := features.dat
SCRIPTDIR := $(CURDIR)/../scripts
BUILDDIR := $(CURDIR)/../../build
LABELS := pop:rock:classical:electronic:rap

FILES = $(wildcard *.mp3)

all: features

%.lfm: %.mp3 $(SCRIPTDIR)/lastfm
	$(SCRIPTDIR)/lastfm "$<" >"$@"

%.tag: %.lfm $(SCRIPTDIR)/tagproc.xsl
	xsltproc -o "$@" "$(SCRIPTDIR)/tagproc.xsl" "$<"

%.wav: %.mp3
	ffmpeg -i "$<" -ac 1 -ss 00:02 "$@"

m_waves: $(FILES:.mp3=.wav)
m_tags: $(FILES:.mp3=.tag)
waves: fixnames m_waves
tags: fixnames m_tags

$(OUT): waves tags
	$(BUILDDIR)/genre dataset -l $(LABELS) -o "$@" -d "$(PWD)"

features: $(OUT)

fixnames:
	for i in *.mp3; do n=`echo "$$i" | tr ' #' __`; [ "$$i" == "$$n" ] || mv "$$i" "$$n"; done

clean_waves:
	rm -f *.wav
clean_lfm:
	rm -f *.lfm
clean_tags:
	rm -f *.tag
clean:
	rm -f $(OUT)
clean_all: clean_waves clean_lfm clean_tags clean

.PHONY: features clean_waves clean_tags clean_lfm clean fixnames waves tags m_waves m_tags

