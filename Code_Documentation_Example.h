//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// EmtCosim.h
//------------------------------------------------------------------------------
//
// This is an open-source header providing tools for interfacing with the PSCAD
// "Communication Fabric". By using these tools, it is possible to run in a
// Co-Simulation environment with PSCAD/EMTDC.
//
// Created By:
// ~~~~~~~~~~~
//   PSCAD Design Group <pscad@hvdc.ca>
//   Manitoba HVDC Research Centre
//   Winnipeg, Manitoba. CANADA
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifndef EMTCOSIM_H_6482547583562
#define EMTCOSIM_H_6482547583562

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Documentation
// --------------------------------------------------------------------------------------------------------------------------------------------------
//
// EmtdcCosimulation_Channel (Structure)
// ==================================================================================================================================================
// Members:
//
//    pReserved      
//
//       Type:          (void*)      
//       Description:   This member is a pointer to an internal data structure, its contents are private, and maintain the inner working of the 
//                      structure
//
// Methods:
//
// --------------------------------------------------------------------------------------------------------------------------------------------------
//
//    double   GetValue
//                      (
//                         EmtdcCosimulation_Channel*       _this
//                      ,  double                           val
//                      ,  int                              index
//                      )
//
//       Parameters:
//       
//          _this
//             Type:            (EmtdcCosimulation_Channel*)
//             Description:     A pointer to the structure being operated on. This parameter should always be the content from which it is called.
//       
//          time
//             Type:             double
//             Description:      The time value for which the parameter is requested. This parameter should be entered in the format of seconds passed
//                               since the start of the simulation.
//       
//          index
//             Type:             int
//             Description:      The index location of the value being requested.
//       
//       Return Value:
//          Type:             double
//          Description:      The value at the specified index for the valid time value on this channel.
//       
//       Remarks:
//          This function will find the correct value and return it in as efficient time as possible. During its execution it may be reading other
//          channel information from the same client or other time values, These values will be cached or discarded depending on the status of the 
//          channel that received the information.
//
//          The time value for the channel MUST increase over time and never decrease. For example if the previous call on this channel had a time
//          value of 0.002 then the next call must be this value or higher. (Values are discarded after they are no longer in the valid time cycle.
//
//          Each Channel is independent in its cached values and time domain use when retrieving values.
//
//          The time-step does not need to be synchronized with the other end of the channel in the Co-Simulation, and the valid value will be used
//          for the time provided independently of the time-step of either application.
//
//          It is recommended that you do not read values at the 0 time, and to start with an initial condition rather than reading from the channel
//          this is to prevent the possibility of dead-locking. Both application should start with initial conditions and run their first time step
//          independently.
//
//          The max size of the index is one less than value returned from GetRecvSize()
//
//       Example:
//          // Get the channel with channel ID of 10
//          EmtdcCosimulation_Channel * channel = EmtdcCosimulation_FindChannel(10);
//          double time = 0.000;
//          double time_step = 0.001;
//
//          while (time < 1 )
//             {
//             double val1 = 0;
//             double val2 = 0;         
//             if ( time > 0 )
//                {
//                channel->GetValue(channel, time, 0);
//                channel->GetValue(channel, time, 1);
//                }
//
//             // Perform Computations
//                .
//                .
//                .
//
//             // Increment the time step
//             time += time_step;
//             }
//
// --------------------------------------------------------------------------------------------------------------------------------------------------
//
//    void  SetValue
//                   (
//                      EmtdcCosimulation_Channel*       _this
//                   ,  double                           value
//                   ,  int                              index
//                   )
//
//       Parameters:
//       
//          _this
//             Type:            (EmtdcCosimulation_Channel*)
//             Description:     A pointer to the structure being operated on. This parameter should always be the content from which it is called.
//       
//          value
//             Type:             double
//             Description:      The value that is to be sent to the other application at the other end of this channel
//
//          index
//             Type:             int
//             Description:      The index of the value to be sent.
//
//       Return Value:
//          No return value
//
//       Remarks:
//          This function sets values in a cache to be sent when a call to 'Send' is made. This function does not send the value explicitly.
//    
//          The max index value is one less than the return value from GetSendSize()
//
//       Example:
//
//          // Get the channel with channel ID of 10
//          EmtdcCosimulation_Channel * channel = EmtdcCosimulation_FindChannel(10);
//          double time = 0.000;
//          double time_step = 0.001;
//
//          while (time < 1 )
//             {
//             // Perform Computations
//                .
//                .
//                .
//       
//             // Set the value to be sent
//             channel->SetValue(channel, val1, 0);
//             channel->SetValue(channel, val2, 1);
//          
//             Send the cached values, they are valid until the next time step is complete
//             channel->Send(time + time_step);
//
//             // Increment the time step
//             time += time_step;
//             }
// 
// --------------------------------------------------------------------------------------------------------------------------------------------------
//
//    void  Send
//                   (
//                      EmtdcCosimulation_Channel*       _this
//                      double                           time
//                   )
//
//       Parameters:
//       
//          _this
//             Type:            (EmtdcCosimulation_Channel*)
//             Description:     A pointer to the structure being operated on. This parameter should always be the content from which it is called.
//       
//          time
//             Type:             double
//             Description:      The time value for the values sent will be valid until.
//
//       Return Value:
//          No return value
//
//       Remarks:
//          Values are sent with this call, use SetValue() to set the values that will be sent. (any unset value are defaulted to 0)/
//
//          It is assumed that value validity does not overlap between messages. If the previous send was 0.001 then this send will not be valid until
//          the 0.001 has passed, then it will be valid until what ever time was passed.
//
//          Each channel is independent in terms of valid time sent.
//
//          Good practice is to alway send first then receive that value required for the next time-step.
//
//       Example:
//
//          // Get the channel with channel ID of 10
//          EmtdcCosimulation_Channel * channel = EmtdcCosimulation_FindChannel(10);
//          double time = 0.000;
//          double time_step = 0.001;
//
//          while (time < 1 )
//             {
//             // Perform Computations
//                .
//                .
//                .
//       
//             // Set the value to be sent
//             channel->SetValue(channel, val1, 0);
//             channel->SetValue(channel, val2, 1);
//          
//             Send the cached values, they are valid until the next time step is complete
//             channel->Send(time + time_step);
//
//             // Increment the time step
//             time += time_step;
//             }
// 
// --------------------------------------------------------------------------------------------------------------------------------------------------
//    unsigned int   GetChannelId
//                   (
//                      EmtdcCosimulation_Channel*       _this
//                   )
//
//       Parameters:
//       
//          _this
//             Type:            (EmtdcCosimulation_Channel*)
//             Description:     A pointer to the structure being operated on. This parameter should always be the content from which it is called.
//       
//       Return Value:
//          Type:             unsigned int
//          Description:      The channel id for this channel
//
//       Remarks:
//          This value will not change for the duration of the channel life.
//
//       Example:
//
//          // Get the channel with channel ID of 10
//          EmtdcCosimulation_Channel * channel = EmtdcCosimulation_FindChannel(10);
//          
//          // Ensure channel id is as expected
//          ASSERT(channel->GetChannelId(channel) == 10);
//
// --------------------------------------------------------------------------------------------------------------------------------------------------
//
//   int    GetSendSize
//                   (
//                      EmtdcCosimulation_Channel* _this
//                   )
//
//       Parameters:
//       
//          _this
//             Type:            (EmtdcCosimulation_Channel*)
//             Description:     A pointer to the structure being operated on. This parameter should always be the content from which it is called.
//       
//       Return Value:
//          Type:             int
//          Description:      The amount of double that will be sent with each send call.
//
//       Remarks:
//          This value will not change for the duration of the channel life.
//       
//       Example:
//
//          // Get the channel with channel ID of 10
//          EmtdcCosimulation_Channel * channel = EmtdcCosimulation_FindChannel(10);
//
//          // Ensure this channel can send the required amount of data
//          ASSERT(channel->GetSendSize(channel) == 2);
//
// --------------------------------------------------------------------------------------------------------------------------------------------------
//
//   int    GetRecvSize
//                   (
//                      EmtdcCosimulation_Channel* _this
//                   )
//
//       Parameters:
//       
//          _this
//             Type:            (EmtdcCosimulation_Channel*)
//             Description:     A pointer to the structure being operated on. This parameter should always be the content from which it is called.
//       
//       Return Value:
//          Type:             int
//          Description:      The amount of double that will be received in a given message.
//
//       Remarks:
//          This value will not change for the duration of the channel life.
//       
//       Example:
//
//          // Get the channel with channel ID of 10
//          EmtdcCosimulation_Channel * channel = EmtdcCosimulation_FindChannel(10);
//
//          // Ensure this channel can send the required amount of data
//          ASSERT(channel->GetRecvSize(channel) == 2);
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Functions:
// ==================================================================================================================================================
//
//    void  EmtdcCosimulation_InitializeCosimulation
//                   (
//                      const char *      fabric_location
//                   ,  const char *      hostname
//                   ,  int               port
//                   ,  int               client_id
//                   )
//
//       Parameters:
//
//          fabric_location
//             Type:          const char *   (String)
//             Description:   The absolute path the Communication Fabric DLL (ComFab.dll)
//
//          hostname          
//             Type:          const char *   (String)
//             Description:   The Host Name or IP Address of the master process (PSCAD) that will coordinate the communication.
//
//          port
//             Type:          int
//             Description:   The port the master process (PSCAD) that will coordinate the communication is listening on.
//
//          client_id
//             Type:          int
//             Description:   A system wide unique ID for this process (it must be between 30000 and 65535) each process in the co-simulation must
//                            have a system wide unique ID.
//
//       Return Value:
//          No return value
//
//       Remarks:
//          This call may take a a finite amount of time to return. During this call all necessary connection are established and the required
//          channels are constructed. The master process coordinates this entire process. This function will block until the entire system is
//          properly configured.
//
//          This function should be call only once per process during per co-simulation.
//
//       Example:
//          // Initialize the Co-Simulation
//          EmtdcCosimulation_InitializeCosimulation("C:\\Program Files (x86)\\CommunicationFabric\\x86\\ComFab.dll", "localhost", 34343, 32000 );
//
// ==================================================================================================================================================
//
//   EmtdcCosimulation_Channel *  EmtdcCosimulation_FindChannel
//                   (
//                      unsigned int channel_id
//                   )
//
//       Parameters:
//
//          channel_id
//             Type:          unsigned int
//             Description:   The Id of the channel being searched for
//
//       Return Value:
//          Type:             EmtdcCosimulation_Channel*
//          Description:      The channel containing the ID specified. (If channel exists with the ID it will return NULL)
//
//       Remarks:
//          Before this can be executed EmtdcCosimulation_InitializeCosimulation must be called
//          If the Co-Simulation is no initialized then it will return NULL
//          If no channel exists in this process with that channel ID then it will return NULL
//
//       Example:
//          // Get the channel with channel ID of 10
//          EmtdcCosimulation_Channel * channel = EmtdcCosimulation_FindChannel(10);
// 
// ==================================================================================================================================================
//    void   EmtdcCosimulation_FinalizeCosimulation
//                   (
//                      void
//                   )
//
//       Parameters:
//          No Parameters
//
//       Return Value:
//          No return value
//
//       Remarks:
//          This function will perform any cleanup required from the co-simulation. All EmtdcCosimulation_Channel will be deallocated after this
//          call. 
//          
//          Note: this call will inform the master process (PSCAD) that the client has finished running, then waits for PSCAD to give the go-ahead
//                to disconnect.
//
//       Example:
//          // Finalize the Co-Simulation
//          EmtdcCosimulation_FinalizeCosimulation();
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifdef __cplusplus
   #define EXTERN
#else
   #define EXTERN extern
#endif

//==============================================================================
// EmtdcCosimulation_Channel
//------------------------------------------------------------------------------
// The manager for a specific channel of information traveling between this
// Client and another client, the channel should be globally unique across the 
// entire system
//==============================================================================
struct EmtdcCosimulation_Channel_S;
typedef struct EmtdcCosimulation_Channel_S EmtdcCosimulation_Channel;

struct EmtdcCosimulation_Channel_S
   {
   void * pReserved;

   double          (*GetValue)     (EmtdcCosimulation_Channel* _this, double time, int index);
   void            (*SetValue)     (EmtdcCosimulation_Channel* _this, double val, int index);
   void            (*Send)         (EmtdcCosimulation_Channel* _this, double time);
   unsigned int    (*GetChannelId) (EmtdcCosimulation_Channel* _this);
   int             (*GetSendSize)  (EmtdcCosimulation_Channel* _this);
   int             (*GetRecvSize)  (EmtdcCosimulation_Channel* _this);
   };

#ifdef __cplusplus
extern "C" {
#endif;

   //==============================================================================
   // InitializeCosimulation
   //------------------------------------------------------------------------------
   // Call this function to start the Co-Simulation Process. Only call this
   // function once per process.
   //==============================================================================
   EXTERN void  EmtdcCosimulation_InitializeCosimulation(const char * fabric_location, const char * hostname, int port, int client_id);

   //==============================================================================
   // FindChannel()
   //------------------------------------------------------------------------------
   // Get the channel with the channel_id specified (if it exists)
   //==============================================================================
   EXTERN EmtdcCosimulation_Channel *   EmtdcCosimulation_FindChannel(unsigned int channel_id);

   //==============================================================================
   // FinalizeCosimulation
   //------------------------------------------------------------------------------
   // Call this function to end the Co-Simulation process
   //==============================================================================
   EXTERN void   EmtdcCosimulation_FinalizeCosimulation();

#ifdef __cplusplus
   }
#endif;

#endif // Header Guard
