# WORKSHOPS := clustering_pca_autoencoders conv_nets LogisticRegression nn_backprop rnn
WORKSHOPS := nn_backprop conv_nets

all:
	# Loop over all workshops and build them.
	$(foreach WORKSHOP, $(WORKSHOPS), \
		echo --------- BUILDING WORKSHOP $(WORKSHOP) -------------- ; \
		make --directory=$(WORKSHOP) WORKSHOP=$(WORKSHOP) all; \
		if [ $$? -eq 0 ]; then \
			echo ---------  BUILDING WORKSHOP $(WORKSHOP) SUCCESS ------; \
		else \
			echo ---------  BUILDING WORKSHOP $(WORKSHOP) FAILED ------; \
			exit 1; \
		fi; \
	)