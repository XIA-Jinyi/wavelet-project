N_TRAIN ?= 350
N_TEST  ?= 150
RATE    ?= 1.0
WAVELET ?= db4

.PHONY: data spearman features train test analyze verify clean

# 1. Data preparation
data:
	cd src && python data_prep.py

# 2. Spearman correlation analysis
spearman:
	cd src && python spearman.py --wavelet $(WAVELET) --rate $(RATE) --n-train $(N_TRAIN)

# 3. Feature extraction (cached)
features:
	cd src && python extract_features.py --wavelet $(WAVELET) --rate $(RATE) --n-train $(N_TRAIN) --n-test $(N_TEST)

# 4. Training
train-wdcnn:
	cd src && python train_wdcnn.py --wavelet $(WAVELET) --rate $(RATE) --n-train $(N_TRAIN)

train-mlp:
	cd src && python train_mlp.py --wavelet $(WAVELET) --rate $(RATE) --n-train $(N_TRAIN)

train-svm:
	cd src && python train_svm.py --wavelet $(WAVELET) --rate $(RATE) --n-train $(N_TRAIN)

train-all:
	for w in haar db4; do \
		for r in 1.0 1.5 2.0; do \
			$(MAKE) features WAVELET=$$w RATE=$$r N_TRAIN=7000 N_TEST=3000; \
			$(MAKE) train-wdcnn WAVELET=$$w RATE=$$r N_TRAIN=7000; \
			$(MAKE) train-mlp WAVELET=$$w RATE=$$r N_TRAIN=7000; \
			$(MAKE) train-svm WAVELET=$$w RATE=$$r N_TRAIN=7000; \
		done; \
	done

# 5. Testing
test:
	cd src && python test.py --wavelet $(WAVELET) --rate $(RATE) --n-train $(N_TRAIN) --n-test $(N_TEST)

test-all:
	cd src && python test.py --wavelet $(WAVELET) --rate $(RATE) --n-train $(N_TRAIN) --n-test $(N_TEST) --models wdcnn mlp svm

# 6. Analysis
analyze:
	cd src && python analyze.py --n-train $(N_TRAIN)

# One-shot small-scale verification
verify: data spearman features
	$(MAKE) train-wdcnn WAVELET=db4 RATE=1.0 N_TRAIN=350
	$(MAKE) train-mlp WAVELET=db4 RATE=1.0 N_TRAIN=350
	$(MAKE) train-svm WAVELET=db4 RATE=1.0 N_TRAIN=350
	$(MAKE) test-all WAVELET=db4 RATE=1.0 N_TRAIN=350 N_TEST=150
	$(MAKE) analyze N_TRAIN=350
	@echo "Verification complete. See output/figures/"

clean:
	rm -rf model/* output/features/* output/results/* output/figures/*
	@echo "Cleaned model/ and output/"
