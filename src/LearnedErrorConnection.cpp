/* 
* Copyright 2014-2018 Friedemann Zenke
*
* This file is part of Auryn, a simulation package for plastic
* spiking neural networks.
* 
* Auryn is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* Auryn is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with Auryn.  If not, see <http://www.gnu.org/licenses/>.
*
* If you are using Auryn or parts of it for your work please cite:
* Zenke, F. and Gerstner, W., 2014. Limits to high-speed simulations 
* of spiking neural networks using general-purpose computers. 
* Front Neuroinform 8, 76. doi: 10.3389/fninf.2014.00076
*/

#include "LearnedErrorConnection.h"
#include "JIafPscExpGroup.h"

using namespace auryn;

LearnedErrorConnection::LearnedErrorConnection( 
		SpikingGroup * source, 
		NeuronGroup * destination, 
		AurynWeight weight, 
		AurynFloat sparseness, 
		TransmitterType transmitter, 
		std::string name) 
	: SparseConnection( 
		source, 
		destination, 
		weight, 
		sparseness, 
		transmitter, 
		name) 
{

	logger->verbose("Initialzing LearnedErrorConnection");
	state_watcher = new StateWatcherGroup(src, "err");
	connect_states("err", "err_in");

	set_min_weight(-1e42); // whatever
	set_max_weight(1e42);

	rdd_params = new ComplexMatrix <float> (get_m_rows(), get_n_cols(), 256, 4);
	rdd_params->fill_zeros();

	rdd_feedback = new ComplexMatrix <float> (get_m_rows(), get_n_cols(), 256, 1);
	rdd_feedback->fill_zeros();

	fb_lr = 0.00001;
}

LearnedErrorConnection::~LearnedErrorConnection()
{
}

void LearnedErrorConnection::connect_states(string pre_name, string post_name)
{
	state_watcher->watch(src, pre_name);
	set_target(post_name);
}

void LearnedErrorConnection::propagate()
{
	if ( dst->evolve_locally() ) { // necessary 
		NeuronID * ind = w->get_row_begin(0); // first element of index array
		AurynWeight * data = w->get_data_begin(); // first element of data array

		// loop over spikes
		for (NeuronID i = 0 ; i < state_watcher->get_spikes()->size() ; ++i ) {
			// get spike at pos i in SpikeContainer
			const NeuronID spike = state_watcher->get_spikes()->at(i);

			// extract spike attribute from attribute stack;
			const NeuronID stackpos = i + (spike_attribute_offset)*src->get_spikes()->size();
			const AurynFloat attribute = state_watcher->get_attributes()->at(stackpos);

			// std::cout << spike << " " << attribute <<  std::endl;
			// return;

			// loop over postsynaptic targets
			for (NeuronID * c = w->get_row_begin(spike) ; 
					c != w->get_row_end(spike) ; 
					++c ) {
				AurynWeight value = data[c-ind] * attribute; 
				transmit( *c , value );
				// transmit( w->get_colind(c) , w->get_value(c) );
			}
		}
	}
}

void LearnedErrorConnection::evolve()
{
	// columns: hidden units
	// rows: output units

	JIafPscExpGroup * dest = (JIafPscExpGroup *) dst;

	// get RDD times for each post-synaptic neuron
	unsigned short * t_rdd = dest->t_rdd;

	for (AurynLong j = 0 ; j < get_n_cols() ; ++j ) {

		// only update feedback weights if RDD window is ending
		if (t_rdd[j]==1) {
		 	// see if the post-synaptic neuron spiked
			AurynFloat max_drive = (dest->t_max_drive)[j];

			for (AurynLong i = 0 ; i < get_m_rows() ; ++i ) {

				AurynLong index = w->get_data_index(i, j);

				if (max_drive > dest->thr) {
					// post-synaptic neuron spiked
					AurynFloat error_term = (rdd_params->get_element(index, 2))*max_drive + rdd_params->get_element(index, 0) - rdd_feedback->get(i, j);

					rdd_params->set_element(index, rdd_params->get_element(index, 2)-fb_lr*error_term*max_drive, 2);
					rdd_params->set_element(index, rdd_params->get_element(index, 0)-fb_lr*error_term, 0);
				} else {
					// post-synaptic neuron did not spike
					AurynFloat error_term = (rdd_params->get_element(index, 3))*max_drive + rdd_params->get_element(index, 1) - rdd_feedback->get(i, j);

					rdd_params->set_element(index, rdd_params->get_element(index, 3)-fb_lr*error_term*max_drive, 3);
					rdd_params->set_element(index, rdd_params->get_element(index, 1)-fb_lr*error_term, 1);
				}

				AurynFloat beta = (rdd_params->get_element(index, 2))*max_drive + rdd_params->get_element(index, 0) - (rdd_params->get_element(index, 3))*max_drive + rdd_params->get_element(index, 1);

				AurynWeight w_val = w->get_data(index);
				w->set(i, j, w_val + 0.01*(beta - w_val));

				// if (index == 100) {
				// std::cout << beta << " \n";
				// }
			}

			rdd_feedback->set_col(j, 0);
		} else if ((t_rdd[j]) > 0) {

			NeuronID * ind = w->get_row_begin(0);

			// add feedback
			for (NeuronID i = 0 ; i < state_watcher->get_spikes()->size() ; ++i ) {
				const NeuronID spike = state_watcher->get_spikes()->at(i);

				AurynFloat rdd_val = rdd_feedback->get(spike, j);

				rdd_feedback->set(spike, j, rdd_val + 1);
	 		}
		}
	}

	// counter++;

	// std::cout << state_watcher->get_spikes()->size() << " ";

	// std::cout << " ";
	// for (AurynLong i = 0 ; i < w->get_nonzero() ; ++i ) {

	// 	for (NeuronID * c = w->get_row_begin(spike) ; 
	// 			c != w->get_row_end(spike) ; 
	// 			++c ) {

	//     AurynWeight weight = w->get_data(i);
	//     // do something with the weight
	//     weight = 0;
	//     w->set(0, 0, 0);
	//     std::cout << w->get(0, 1);
	// }


	// w->set(0, 0, 0);
	// std::cout << w->get(0, 1);
}
