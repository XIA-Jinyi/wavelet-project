N_TRAIN ?= 500
N_TEST  ?= 200
RATE    ?= 0.6
WAVELET ?= db4

.PHONY: data spearman features features-raw features-all train-wdcnn train-mlp train-svm train-cnn train-all test test-all analyze verify clean clean-all

# 1. Data preparation
data:
	cd src && python data_prep.py

# 2. Spearman correlation analysis
spearman:
	cd src && python spearman.py --wavelet $(WAVELET) --rate $(RATE) --n-train $(N_TRAIN)

# 3. Feature extraction (cached)
features:
	cd src && python extract_features.py --wavelet $(WAVELET) --rate $(RATE) --n-train $(N_TRAIN) --n-test $(N_TEST)

features-raw:
	cd src && python extract_features.py --raw-only --n-train $(N_TRAIN) --n-test $(N_TEST) --rate $(RATE)

features-all:
	for r in 0.2 0.4 0.6 0.8 1.0; do \
		for w in haar db4; do \
			$(MAKE) features WAVELET=$$w RATE=$$r; \
		done; \
		$(MAKE) features-raw RATE=$$r; \
	done

# 4. Training
train-wdcnn:
	cd src && python train_wdcnn.py --wavelet $(WAVELET) --rate $(RATE) --n-train $(N_TRAIN)

train-mlp:
	cd src && python train_mlp.py --wavelet $(WAVELET) --rate $(RATE) --n-train $(N_TRAIN)

train-svm:
	cd src && python train_svm.py --wavelet $(WAVELET) --rate $(RATE) --n-train $(N_TRAIN)

train-cnn:
	cd src && python train_cnn.py --rate $(RATE) --n-train $(N_TRAIN)

train-all:
	for r in 0.2 0.4 0.6 0.8 1.0; do \
		for w in haar db4; do \
			$(MAKE) train-wdcnn WAVELET=$$w RATE=$$r; \
			$(MAKE) train-mlp WAVELET=$$w RATE=$$r; \
			$(MAKE) train-svm WAVELET=$$w RATE=$$r; \
		done; \
		$(MAKE) train-cnn RATE=$$r; \
	done

# 5. Testing
test:
	cd src && python test.py --wavelet $(WAVELET) --rate $(RATE) --n-train $(N_TRAIN) --n-test $(N_TEST)

test-all:
	for r in 0.2 0.4 0.6 0.8 1.0; do \
		for w in haar db4; do \
			$(MAKE) test WAVELET=$$w RATE=$$r; \
		done; \
	done

# 6. Analysis
analyze:
	cd src && python analyze.py --n-train $(N_TRAIN)

clean:
	rm -rf output/features/* output/results/* output/figures/*
	@echo "Cleaned output/"

clean-all: clean
	rm -rf model/*
	@echo "Cleaned model/"
