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
#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace auryn;

std::ofstream rdd_param_0_file;
std::ofstream rdd_param_1_file;
std::ofstream rdd_param_2_file;
std::ofstream rdd_param_3_file;
std::ofstream rdd_feedback_file;
std::ofstream max_drive_file;
std::ofstream beta_file;
std::ofstream y_file;

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

	rdd_param_0 = new AurynVector< float > (get_m_rows()*get_n_cols());
	// rdd_param_0->set_all(1.0);
	rdd_param_1 = new AurynVector< float > (get_m_rows()*get_n_cols());
	// rdd_param_1->set_all(1.0);
	rdd_param_2 = new AurynVector< float > (get_m_rows()*get_n_cols());
	rdd_param_3 = new AurynVector< float > (get_m_rows()*get_n_cols());
	rdd_feedback = new AurynVector< float > (get_m_rows()*get_n_cols());
	betas = new AurynVector< float > (get_m_rows()*get_n_cols());

	fb_lr = 0.005;

	learning_active = false;

	rdd_param_0_file.open("rdd_param_0.txt");
	rdd_param_1_file.open("rdd_param_1.txt");
	rdd_param_2_file.open("rdd_param_2.txt");
	rdd_param_3_file.open("rdd_param_3.txt");
	rdd_feedback_file.open("rdd_feedback.txt");
	max_drive_file.open("max_drive.txt");
	beta_file.open("beta.txt");
	y_file.open("y.txt");
	rdd_param_0_file << "";
	rdd_param_1_file << "";
	rdd_param_2_file << "";
	rdd_param_3_file << "";
	rdd_feedback_file << "";
	max_drive_file << "";
	beta_file << "";
	y_file << "";
	rdd_param_0_file.close();
	rdd_param_1_file.close();
	rdd_param_2_file.close();
	rdd_param_3_file.close();
	rdd_feedback_file.close();
	max_drive_file.close();
	beta_file.close();
	y_file.close();

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

	for (NeuronID j = 0 ; j < get_n_cols() ; ++j ) {

		// only update feedback weights if RDD window is ending
		if (t_rdd[j]==1) {
		 	// see if the post-synaptic neuron spiked
			AurynFloat max_drive = (dest->t_max_drive)[j];

			for (NeuronID i = 0 ; i < get_m_rows() ; ++i ) {

				int ind = i*get_n_cols() + j;

				AurynLong index = w->get_data_index(i, j);

				// if (max_drive > dest->thr && max_drive < dest->thr + 0.001) {
				if (max_drive > dest->thr) {
					// post-synaptic neuron spiked
					AurynFloat error_term = (rdd_param_2->get(ind))*(max_drive - dest->thr) + rdd_param_0->get(ind) - rdd_feedback->get(ind);

					// std::cout << max_drive*1000 << "\n";
					// rdd_param_2->set(ind, rdd_param_2->get(ind)-exp(-abs(max_drive - dest->thr)/0.01)*fb_lr*error_term*max_drive);
					// rdd_param_0->set(ind, rdd_param_0->get(ind)-exp(-abs(max_drive - dest->thr)/0.01)*fb_lr*error_term);

					rdd_param_2->set(ind, rdd_param_2->get(ind)-fb_lr*error_term*(max_drive - dest->thr));
					rdd_param_0->set(ind, rdd_param_0->get(ind)-fb_lr*error_term);

					// if (rdd_feedback->get(ind) == 0) {
					// 	rdd_param_0->set(ind, rdd_param_0->get(ind)-0.01);
					// } else {
					// 	rdd_param_0->set(ind, rdd_param_0->get(ind)+0.01);
					// }

				// } else if (max_drive > dest->thr - 0.001) {
				} else {
					// post-synaptic neuron did not spike
					AurynFloat error_term = (rdd_param_3->get(ind))*(max_drive - dest->thr) + rdd_param_1->get(ind) - rdd_feedback->get(ind);

					// rdd_param_3->set(ind, rdd_param_3->get(ind)-exp(-abs(max_drive - dest->thr)/0.01)*fb_lr*error_term*max_drive);
					// rdd_param_1->set(ind, rdd_param_1->get(ind)-exp(-abs(max_drive - dest->thr)/0.01)*fb_lr*error_term);

					rdd_param_3->set(ind, rdd_param_3->get(ind)-fb_lr*error_term*(max_drive - dest->thr));
					rdd_param_1->set(ind, rdd_param_1->get(ind)-fb_lr*error_term);

					// if (rdd_feedback->get(ind) == 0) {
					// 	rdd_param_0->set(ind, rdd_param_0->get(ind)+0.01);
					// } else {
					// 	rdd_param_0->set(ind, rdd_param_0->get(ind)-0.01);
					// }
				}

				AurynFloat beta = rdd_param_0->get(ind) - (rdd_param_1->get(ind));
				betas->set(ind, beta);

				AurynWeight w_val = w->get(i, j);

				int w_sign = (w_val > 0) - (w_val <= 0);
				int beta_sign = (beta > 0) - (beta <= 0);

				if (learning_active) {
					w->set(i, j, w_val + 0.01*(beta - w_val));
				}

				if (i == 1 && j == 0) {
					rdd_param_0_file.open("rdd_param_0.txt", std::ios_base::app);
					rdd_param_1_file.open("rdd_param_1.txt", std::ios_base::app);
					rdd_param_2_file.open("rdd_param_2.txt", std::ios_base::app);
					rdd_param_3_file.open("rdd_param_3.txt", std::ios_base::app);
					rdd_feedback_file.open("rdd_feedback.txt", std::ios_base::app);
					max_drive_file.open("max_drive.txt", std::ios_base::app);
					beta_file.open("beta.txt", std::ios_base::app);
					y_file.open("y.txt", std::ios_base::app);
					rdd_param_0_file << rdd_param_0->get(ind) << "\n";
					rdd_param_1_file << rdd_param_1->get(ind) << "\n";
					rdd_param_2_file << rdd_param_2->get(ind) << "\n";
					rdd_param_3_file << rdd_param_3->get(ind) << "\n";
					rdd_feedback_file << rdd_feedback->get(ind) << "\n";
					max_drive_file << max_drive << "\n";
					beta_file << beta << "\n";
					y_file << w->get(i, j) << "\n";
					rdd_param_0_file.close();
					rdd_param_1_file.close();
					rdd_param_2_file.close();
					rdd_param_3_file.close();
					rdd_feedback_file.close();
					max_drive_file.close();
					beta_file.close();
					y_file.close();
				}

				rdd_feedback->set(ind, 0);
			}
		} else if ((t_rdd[j]) > 0) {
			// add feedback
			for (NeuronID i = 0 ; i < state_watcher->get_spikes()->size() ; ++i ) {
				const NeuronID spike = state_watcher->get_spikes()->at(i);

				int ind = spike*get_n_cols() + j;

				AurynFloat rdd_val = rdd_feedback->get(ind);

				rdd_feedback->set(ind, rdd_val + 1.0*(t_rdd[j]/(float)(dest->rdd_time)));
	 		}
		}
	}
}
