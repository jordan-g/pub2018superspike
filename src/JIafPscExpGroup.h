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

#ifndef JIAFPSCEXPGROUP_H_
#define JIAFPSCEXPGROUP_H_

#include "auryn/auryn_definitions.h"
#include "auryn/AurynVector.h"
#include "auryn/NeuronGroup.h"
#include "auryn/ExpCubaSynapse.h"
#include "auryn/System.h"

namespace auryn {

/*! \brief Simple LIF neuron model with absolute refractoriness and current based synapses 
 *
 * This model is simular to the one used in Vogels and Abbott 2005, with exponential currents and 
 * does not make a distinction between the timecourse of exc and inh synases.
 *
 * To give input to these neurons use set_transmitter("syn_current") as a transmitter target.
 *
 */
class JIafPscExpGroup : public NeuronGroup
{
private:

	unsigned short refractory_time;

	AurynFloat tau_mem, r_mem, c_mem;
	AurynFloat tau_syn;
	AurynFloat scale_mem;

	AurynFloat * t_mem; 

	void init();
	void calculate_scale_constants();
	inline void integrate_state();
	inline void check_thresholds();
	virtual string get_output_line(NeuronID i);
	virtual void load_input_line(NeuronID i, const char * buf);

	void virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version );
	void virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version );
public:

	AurynVector<AurynFloat>  * drive;

	unsigned short * t_rdd;

	unsigned short rdd_time;

	AurynVector<AurynFloat> * max_drive;
	AurynFloat * t_max_drive;
	AurynFloat * t_drive;
	unsigned short * t_ref;

	AurynFloat e_rest,thr;

	AurynFloat rdd_window;

	/*! \brief Vector holding neuronspecific background currents */
	AurynStateVector * bg_current;

	/*! \brief Vector holding neuronspecific synaptic currents */
	AurynStateVector * syn_current;

	ExpCubaSynapse * synapse_model;

	/*! \brief Temp vector */
	AurynStateVector * temp;
	AurynVector<AurynFloat> * temp_2;

	/*! \brief Vector holding neuronspecific state of refractory period */
	AurynVector<unsigned short> * ref;

	AurynVector<unsigned short> * rdd;

	/*! \brief The default constructor of this NeuronGroup */
	JIafPscExpGroup(NeuronID size);

	/*! \brief The default destructor */
	virtual ~JIafPscExpGroup();

	/*! \brief Setter for refractory time [s] */
	void set_refractory_period(AurynDouble t);

	void set_rdd_period(AurynDouble t);

	/*! \brief Sets the membrane time constant (default 20ms) */
	void set_tau_mem(AurynFloat taum);

	/*! \brief Sets the membrane resistance (default 100 M-ohm) */
	void set_r_mem(AurynFloat rm);

	/*! \brief Sets the membrane capacitance (default 200pF) */
	void set_c_mem(AurynFloat cm);

	/*! \brief Sets the exponential time constant for current based synapses */
	void set_tau_syn(AurynFloat tau);

	/*! \brief Resets all neurons to defined and identical initial state. */
	void clear();

	/*! \brief Integrates the NeuronGroup state
	 *
	 * The evolve method internally used by System. */
	void evolve();
};

}

#endif /*IAFPSCEXPGROUP_H_*/

