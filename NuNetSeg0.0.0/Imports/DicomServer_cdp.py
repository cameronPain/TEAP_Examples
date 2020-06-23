#!/usr/bin/env python3
#
#
# The architecture of this code:
# On C_Store event:
#   Save the incomming dicom file to a local directory
#
# On connection closed:
#   We have the data and the connection is closed. We can load and do the processing within the OnConnectionClosed function. Note we cannot os.popen into another window as the NeuralNetworks need to be ran in the __main__ instance. Once this is ran, the server should go back to a state of listening.
#
# cdp 20200404
#
# cdp 20200405 I upgraded to the developer version of pynetdicom (1.5.0.dev0) as this has the ability to include additional args into the callback functions. This will allow me to define a NeuNetProcessing class and have access to the information sent between AETitles in the association. The previous verison pynetdicom 1.4.1 did not have this. The server runs on a different thread, so you would need to load your tensorflow models each time an association is made which would take an additional 60 seconds so I want to avoid this.
#
#
#
from pydicom import dcmread, FileDataset, Dataset
import pydicom
from pydicom.uid import ImplicitVRLittleEndian, JPEGBaseline
from pynetdicom import AE, evt, StoragePresentationContexts, association, debug_logger, build_context
from pynetdicom.sop_class import NuclearMedicineImageStorage, CTImageStorage, RTStructureSetStorage, VerificationSOPClass, SecondaryCaptureImageStorage
from pynetdicom.association import Association
import pynetdicom
import os
import time
pnd_version = pynetdicom.__version__.split('.')
if int(pnd_version[0]) < 1 or int(pnd_version[1]) < 5:
    print('ERROR: Running pynetdicom.__version__ = ' + pynetdicom.__version__ + ' Upgrade to 1.5.0.dev0 \n pip install git+https://github.com/pydicom/pynetdicom.git')
    exit()

class DicomServer:
    def __init__(self, ae_title, ip, port, temp, debug = False):
        self.pynetdicom_version = pynetdicom.__version__
        if int(pnd_version[0]) < 1 or int(pnd_version[1]) < 5:
            print('ERROR: Running pynetdicom.__version__ = ' + pynetdicom.__version__ + ' Upgrade to 1.5.0.dev0 \n pip install git+https://github.com/pydicom/pynetdicom.git')
            exit()
        if temp[-1] != '/':
            temp += '/'
        else:
            pass
        self.temp = temp
        try:
            os.popen('mkdir ' + self.temp);
        except:
            os.popen('rm -rf ' + self.temp);
            time.sleep(0.5);
            os.popen('mkdir ' + self.temp);
        self.port = port
        self.ip   = ip
        self.ae   = AE(ae_title   = ae_title)
        self.ae.add_supported_context(NuclearMedicineImageStorage)
        self.ae.add_supported_context(CTImageStorage)
        
        #Specify the functions to call on dicom communication as the defaults.
        self.debug                                 = debug
        self.HandlerFunctions                      = [self.DefOnDicomEcho,
                                                      self.DefOnReceiveStore,
                                                      self.DefOnAssociationRelease,
                                                      self.DefOnAssociationEstablished,
                                                      self.DefOnAssociationAborted,
                                                      self.DefOnAssociationAccepted,
                                                      self.DefOnACSE_Received_Primitive_From_DUL_SP,
                                                      self.DefOnACSE_Sent_Primitive_To_SUL_SP,
                                                      self.DefOnConnectionClosed,
                                                      self.DefOnConnectionOpened,
                                                      self.DefOnDataReceived,
                                                      self.DefOnDataSent,
                                                      self.DefOnDIMSE_Service_Recieved_Decoded_Message,
                                                      self.DefOnDIMSE_Service_Sent_Encoded_Message,
                                                      self.DefOnState_Machine_Transition,
                                                      self.DefOnPDU_Received_From_Peer,
                                                      self.DefOnPDU_Sent_To_Peer,
                                                      self.DefOnAssociationRejected,
                                                      self.DefOnAssociationRequested ]
        self.HandlerFunctionArgs                   = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        self.EventsHandled                         = [evt.EVT_C_ECHO,        #0
                                                      evt.EVT_C_STORE,       #1
                                                      evt.EVT_RELEASED,      #2
                                                      evt.EVT_ESTABLISHED,   #3
                                                      evt.EVT_ABORTED,       #4
                                                      evt.EVT_ACCEPTED,      #5
                                                      evt.EVT_ACSE_RECV,     #6
                                                      evt.EVT_ACSE_SENT,     #7
                                                      evt.EVT_CONN_CLOSE,    #8
                                                      evt.EVT_CONN_OPEN,     #9
                                                      evt.EVT_DATA_RECV,     #10
                                                      evt.EVT_DATA_SENT,     #11
                                                      evt.EVT_DIMSE_RECV,    #12
                                                      evt.EVT_DIMSE_SENT,    #13
                                                      evt.EVT_FSM_TRANSITION,#14
                                                      evt.EVT_PDU_RECV,      #15
                                                      evt.EVT_PDU_SENT,      #16
                                                      evt.EVT_REJECTED,      #17
                                                      evt.EVT_REQUESTED]     #18
        

            # defines what event to look for (C_STORE) and what to do with it (OnReceiveStore).
        self.send_ae = AE()
        self.send_ae.add_requested_context(VerificationSOPClass)
        self.send_ae.add_requested_context(SecondaryCaptureImageStorage)
        self.send_ae.add_requested_context(RTStructureSetStorage, ImplicitVRLittleEndian)
    
    #-------------------------------------------------#
    #        Default Dicom Communication Methods      #
    #-------------------------------------------------#
    def DefOnReceiveStore(self, event, debug = False): #Save the incoming dicom into a temp folder.
        if debug:
            print('Receive store event')
        return 0x0000   #Return the hexadecimal 0 which is interpereted as success in DICOM talk.

    def DefOnConnectionClosed(self, event, debug = False):
        if debug:
            print('Connection closed.')
        return 0x0000

    def DefOnAssociationAborted(self, event, debug = False):
        if debug:
            print('Association aborted.')
        return 0x0000
    
    def DefOnAssociationAccepted(self, event, debug = False):
        if debug:
            print('Association accepted.')
        return 0x0000
    
    def DefOnACSE_Received_Primitive_From_DUL_SP(self, event, debug = False):
        if debug:
            print('ACSE received a primitive from the DUL service provider.')
        return 0x0000
    
    def DefOnACSE_Sent_Primitive_To_SUL_SP(self, event, debug = False):
        if debug:
            print('ACSE sent a primitive to the DUL service provider.')
        return 0x0000
    
    def DefOnConnectionOpened(self, event, debug = False):
        if debug:
            print('Connection opened.')
        return 0x0000
    
    def DefOnDataReceived(self, event, debug = False):
        if debug:
            print('Data recieved.')
        return 0x0000
    
    def DefOnDataSent(self, event, debug = False):
        if debug:
            print('Data sent.')
        return 0x0000
    
    def DefOnDIMSE_Service_Recieved_Decoded_Message(self, event, debug = False):
        if debug:
            print('DIMSE service received and decoded a message.')
        return 0x0000
    
    def DefOnDIMSE_Service_Sent_Encoded_Message(self, event, debug = False):
        if debug:
            print('DIMSE service encoded and sent a message.')
        return 0x0000
    
    def DefOnState_Machine_Transition(self, event, debug = False):
        if debug:
            print('State machine transitioning.')
        return 0x0000
    
    def DefOnPDU_Received_From_Peer(self, event, debug = False):
        if debug:
            print('PDU received from the peer AE.')
        return 0x0000
    
    def DefOnPDU_Sent_To_Peer(self, event, debug = False):
        if debug:
            print('PDU sent to the peer AE.')
        return 0x0000
    
    def DefOnAssociationRejected(self, event, debug = False):
        if debug:
            print('Association rejected.')
        return 0x0000
    
    def DefOnAssociationRequested(self, event, debug = False):
        if debug:
            print('Association requested.')
        return 0x0000
    
    def DefOnAssociationEstablished(self, event, debug = False):
        if debug:
            print('Association established.')
        return 0x0000
    
    def DefOnAssociationRelease(self, event, debug = False):
        if debug:
            print('Association released.')
        return 0x0000
    
    def DefOnDicomEcho(self, event, debug = False):
        if debug:
            print('Echo received.')
        return 0x0000
#-------------------------------------------------#
#                    METHODS                      #
#-------------------------------------------------#
    #Start the server listening.
    def start_server(self, NeuNetSeg):
        #Create a default handlersStore.
        print('building working directory.')
        os.popen('rm -rf ' + NeuNetSeg.settings.temp_directory)
        time.sleep(0.1)
        os.popen('mkdir ' + NeuNetSeg.settings.temp_directory)
        time.sleep(0.1)
        print('compiling event handler store.')
        self.handlersStore      = []
        for i in range(len(self.EventsHandled)):
            self.handlersStore.append((self.EventsHandled[i], self.HandlerFunctions[i], self.HandlerFunctionArgs[i]))
        print('starting server...')
        self.ae.start_server((self.ip, self.port), evt_handlers = self.handlersStore)
    
    #Kill the listening server.
    def shutdown(self):
        self.ae.shutdown()

    #Send a locally stored dicom file to a specified location.
    def send(self, dicom_file_path,rec_ae_title, rec_ip, rec_port):
        print('     Reading saved RT structure.')
        ds = pydicom.read_file(dicom_file_path, force = True)
        print('     Requesting association with receiving entity.')
        assoc = self.send_ae.associate(rec_ip, rec_port, ae_title=rec_ae_title)
        if assoc.is_established:
            Sent = True
            print('Association established!')
            print('Sending file...')
            send_status = assoc.send_c_store(ds)
            print('Releasing association...')
            assoc.release()
            print('Association released.')
        else:
            Sent = False
            print('Could not establish association.')
        print('Done.')

    #Specify a self defined function you want to execute on a specific dicom communication event.
    def specify_dicom_communication_function(self, event_type, function, args = []): #The event type is the pynetdicom.evt.EVENT_TYPE object and function is what you want to do. The first argument of function needs to be the event.
        if event_type in self.EventsHandled:
            print('Updating the handler functions')
            idx = self.EventsHandled.index(event_type)
            self.HandlerFunctions[idx]    = function
            self.HandlerFunctionArgs[idx] = args
        else:
            print('appending event to handlsersStore')
            self.EventsHandled.append(event_type)
            self.HandlerFunctions.append(function)
        #void

def main(ae_title, ip, port, debug = False):
    dcm_server = DicomServer(ae_title, ip, port, debug = debug)
    test_evt = evt.EVT_C_STORE
    dcm_server.specify_dicom_communication_function(test_evt, OnReceiveStore)
    dcm_server.start_server()




import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ae_title', type = str, help = 'The AE title for the server.')
    parser.add_argument('ip'      , type = str, help = 'The ip address you want to run the server on.')
    parser.add_argument('port'    , type = int, help = 'The port number you want to run the server on.')
    parser.add_argument('--debug' , dest = 'debug', default = False, action = 'store_true', help = 'Set this to look at print statements.')
    args = parser.parse_args()
    main(args.ae_title, args.ip, args.port, args.debug)












