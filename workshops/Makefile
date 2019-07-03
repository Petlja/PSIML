# WORKSHOPS := clustering_pca_autoencoders conv_nets LogisticRegression nn_backprop rnn
WORKSHOPS := nn_backprop conv_nets intro_to_tensorflow

all:
	# Loop over all workshops and build them.
	$(foreach WORKSHOP, $(WORKSHOPS), \
		@echo --------- BUILDING WORKSHOP $(WORKSHOP) -------------- ; \
		make --directory=$(WORKSHOP) WORKSHOP=$(WORKSHOP) all ; \
		echo ---------  BUILDING WORKSHOP $(WORKSHOP) SUCCESS ------ ; \
	)