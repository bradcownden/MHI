
!=======================================================================
!> Firing of IGBTs in a H-Bridge cell
!
      SUBROUTINE MMC_Firing_FullBridge(Ncells,Idx,Dim,Iac,BlkIn,T1,T2,T3,T4,Clk)
!
!*************************************************************************************
! SUBROUTINE FOR THE FIRING OF IGBTs IN A H-BRIDGE CELL.
! NOTE: IT RE-USES THE HALF CELL SUBROUTINE AND MANIPULATES THE OUTPUTS OF SUCH
! SUBROUTINE SUCH THAT PULSES FOR THE FOUR IGBT'S ARE ISSUED AS REQUIRED
!*************************************************************************************
!
! Include Files
! -------------
      include 'emtstor.h'
      include 's1.h'
      
      ! The possible states in the H-Bridge are:
      ! 1. Insert positive voltage, Ncells>0. Fire (T1 & T4)
      ! 2. Insert negative voltage, Ncells<0. Fire (T2 & T3)
      ! 3. Bypass cell. Fire (T1 & T3) or Fire (T2 & T4)
      
!
! Arguments
! --------
      INTEGER,INTENT(IN)  :: Ncells  !< Required number of inserted cells
      INTEGER,INTENT(IN)  :: Idx     !< Vector Idx(Dim) with the index for lowest to highest capacitor voltage
      INTEGER,INTENT(IN)  :: Dim     !< Number of cells
      REAL   ,INTENT(IN)  :: Iac     !< Current entering the top of the Half cell bridge,
                                     !! Sign convention: Positive current charges the capacitor when 1 & 4 are on
      INTEGER,INTENT(IN)  :: BlkIn   !< State: 0 for no blocking, 1 for blocked state
      INTEGER,INTENT(OUT) :: T1      !< Firing orders for IGBT 1
      INTEGER,INTENT(OUT) :: T2      !< Firing orders for IGBT 2
      INTEGER,INTENT(OUT) :: T3      !< Firing orders for IGBT 3
      INTEGER,INTENT(OUT) :: T4      !< Firing orders for IGBT 4
      INTEGER,INTENT(IN)  :: Clk     !< If Clk=-1, update firing pulses when Ncells changes
                                     !cont: if Clk=0 or 1, then update firing given clock pulse between 0 and 1
! 
! Local variables
! ---------------
      INTEGER Nold    ! Old number of cells
      INTEGER Blk     ! Local storage for Block
      INTEGER BlkOld
      INTEGER k1,k2,n ! indices
      INTEGER CellSt  ! Cell state -1:-Vc, 0:Bypassed, 1:+Vc, 2:Blocked
      INTEGER deltaN,dk,StartX,FinishX
      INTEGER NewSt,Flg
!
      DIMENSION Idx(Dim),T1(Dim),T2(Dim),T3(Dim),T4(Dim),CellSt(Dim)
      
      ! Possible states or the Full bridge:
      ! 1. Insert positive +Vc
      !-1. Insert negative -Vc
      ! 0. Bypass capacitor
      ! 2. Blocked cell: not inserted nor bypassed - all IGBTs in off state
      
      ! Storage allocations:
      ! NSTORI            : Ncells
      ! NSTORI        +k1 : T1
      ! NSTORI +  Dim +k1 : T2
      ! NSTORI +2*Dim +k1 : T3
      ! NSTORI +3*Dim +k1 : T4
      ! NSTORI +4*Dim +k1 : CellSt --> Cell state -1:-Vc, 0:Bypassed, 1:+Vc, 2:Blocked
      ! NSTORI +5*Dim +1  : Block state for ALL cells 0:No, 1:Blocked
                   

      IF(TIMEZERO) THEN
          ! Initializes all the cells in blocked state (T1=0, T2=0, T3=0, T4=0)
!         STORI(NSTORI) = 0
          STORI(NSTORI+5*Dim+1) = 1
          DO k1 = 1,Dim
!             STORI(NSTORI +       k1) = 0
!             STORI(NSTORI +  Dim +k1) = 0
!             STORI(NSTORI +2*Dim +k1) = 0
!             STORI(NSTORI +3*Dim +k1) = 0
              STORI(NSTORI +4*Dim +k1) = 2
          ENDDO
      ENDIF
     
      ! Retrieves stored information
      Nold   = STORI(NSTORI)
      BlkOld = STORI(NSTORI+5*Dim+1)
      DO k1 = 1,Dim
          T1(k1)     = STORI(NSTORI +       k1)
          T2(k1)     = STORI(NSTORI +  Dim +k1)
          T3(k1)     = STORI(NSTORI +2*Dim +k1)
          T4(k1)     = STORI(NSTORI +3*Dim +k1)
          CellSt(k1) = STORI(NSTORI +4*Dim +k1)
      ENDDO
      
      Blk = BlkIn
        
      ! If it transitions from blocked to unblocked proceed to set all the cells to bypass (N=0)
      ! Also resets Nold, so it will allow for cells to be inserted if the Ncells input calls for it
      IF((Blk .EQ. 0) .AND. (BlkOld .EQ. 1)) THEN
          ! Bypass capacitor, either using (1 & 3) or (2 & 4)
          Nold   = 0
          T1     = 1
          T2     = 0
          T3     = 1
          T4     = 0
          CellSt = 0
      ENDIF
      
      ! If all cells must be blocked 
      IF (Blk .EQ. 1) THEN
          T1     = 0
          T2     = 0
          T3     = 0
          T4     = 0
          CellSt = 2
        
      ELSEIF (Blk .EQ. 0) THEN
          ! Note: if Ncells=Nold it means that there is no change on number of levels needed, therefore do nothing.
          ! If there has been an increment in the number of levels needed then proceed to insert as needed
          
          ! if there has been a change of sign in the number for required inserted cells from +Vc to -Vc (or viceversa)
          ! then reset all cells to bypass state (also resets Nold)
          
          deltaN = Ncells - Nold
          Flg=0
          IF (Clk.Eq.-1) THEN
              IF ((deltaN .NE. 0).OR.(Ncells.EQ.0)) THEN
                  Flg=1
              ENDIF
          ELSEIF (Clk.EQ.1) THEN
              Flg=1
          ENDIF
          IF (Flg.EQ.1) THEN
              ! Bypass all capacitors, either using (1 & 3) or (2 & 4)
              T1     = 1
              T2     = 0
              T3     = 1
              T4     = 0
              CellSt = 0
              ! +++++++++++++++++++++++++++++++
              ! if inserting (+)Vc
              ! +++++++++++++++++++++++++++++++
              IF (Ncells .GT. 0) THEN
                  ! if current negative INSERT (+Vc) capacitors with highest voltage so they discharge
                  IF (Iac .LT. 0.0) THEN
                      NewSt   =  1
                      StartX  =  Dim
                      FinishX =  1
                      dk      = -1
                  ! else--> if current positive then INSERT(+Vc) capacitors with lowest voltage so they charge
                  ELSE
                      NewSt   =  1
                      StartX  =  1
                      FinishX =  Dim
                      dk      =  1
                  ENDIF
              ENDIF
              ! -------------------------------
              ! if inserting (-)Vc
              ! -------------------------------
              IF(Ncells .LT. 0) THEN
                  ! if current positive INSERT (-Vc) capacitors with HIGHEST voltage so they discharge
                  IF(Iac.GT.0.0) THEN
                      NewSt   = -1
                      StartX  =  Dim
                      FinishX =  1
                      dk      = -1 
                  ! else--> if current negative then INSERT(-Vc) capacitors with LOWEST voltage so they charge
                  ELSE
                      NewSt   = -1
                      StartX  =  1
                      FinishX =  Dim
                      dk      =  1 
                  ENDIF
              ENDIF

              IF (Ncells .NE. 0) THEN
                  ! INSERT (-)Vc
                  IF (NewSt .EQ. -1) THEN
                      k1 = 0
                      DO k2 = StartX,FinishX,dk
                          n = Idx(k2)
                          T1(n)     = 0
                          T2(n)     = 1
                          T3(n)     = 1
                          T4(n)     = 0
                          CellSt(n) = -1
                          k1 = k1 + 1              
                          IF(k1 .EQ. ABS(Ncells)) EXIT
                      ENDDO
                  ! INSERT (+)Vc
                  ELSEIF (NewSt .EQ. 1) THEN
                      k1 = 0
                      DO k2 = StartX,FinishX,dk
                          n = Idx(k2)
                          T1(n)     = 1
                          T2(n)     = 0
                          T3(n)     = 0
                          T4(n)     = 1
                          CellSt(n) = 1
                          k1 = k1 + 1             
                          IF(k1 .EQ. ABS(Ncells)) EXIT
                      ENDDO
                  ENDIF
              ENDIF
          ENDIF
      ENDIF  
         
      ! Stores information for next time step
      STORI(NSTORI) = Ncells
      STORI(NSTORI+5*Dim+1) = Blk 
      DO k1 = 1,Dim
         STORI(NSTORI +       k1) = T1(k1)
         STORI(NSTORI +  Dim +k1) = T2(k1)
         STORI(NSTORI +2*Dim +k1) = T3(k1)
         STORI(NSTORI +3*Dim +k1) = T4(k1)
         STORI(NSTORI +4*Dim +k1) = CellSt(k1)
      ENDDO        

      NSTORI = NSTORI + 5*Dim + 2
      
      RETURN
      END SUBROUTINE MMC_Firing_FullBridge
!
!=======================================================================
!> Firing IGBTs in a Half-Cell Bridge: Given an index of capacitor voltages
!! (from lowest to greatest) and the required number of inserted cells, it
!! issues the firing pulses while balancing the capacitor voltages
!
      SUBROUTINE MMC_Firing_HalfBridge(Ncells,Idx,Dim,Iac,Blk,T1,T2)
!
! Include Files
! -------------
      include 'emtstor.h'
      include 's1.h'
      
      ! Description:
      ! This subroutine is written assuming that the the bypass IGBT is at the top of the cell 'T1'
      ! and the insert IGBT is at the bottom of the cell 'T2'      
!
! Arguments
! --------
      INTEGER,INTENT(IN)  :: Ncells  !< Required number of inserted cells
      INTEGER,INTENT(IN)  :: Idx     !< Index for cell voltages from lowest to highest
      INTEGER,INTENT(IN)  :: Dim     !< Number of cells
      REAL   ,INTENT(IN)  :: Iac     !< Current entering the top of the Half cell bridge
      INTEGER,INTENT(IN)  :: Blk     !< State: 0 for no blocking, 1 for blocked state
      INTEGER,INTENT(OUT) :: T1      !< Firing orders for IGBT 1 Top
      INTEGER,INTENT(OUT) :: T2      !< Firing orders for IGBT 2 Bottom
! 
! Local variables
! ---------------
      INTEGER k1,k2,Nnew,Nold,n,BlkOld
      INTEGER CellSt      ! CellSt : 0-->Bypassed, 1-->Inserted, 2-->Blocked
!
      DIMENSION Idx(Dim),T1(Dim),T2(Dim),CellSt(Dim)
      
      ! Storage allocations:
      ! NSTORI            : Ncells
      ! NSTORI        +k1 : T1
      ! NSTORI +  Dim +k1 : T2
      ! NSTORI +2*Dim +k1 : CellSt
      ! NSTORI +3*Dim +1  : Blk
    
      ! Starts all the cells in block state (T1=0, T2=0)
      ! First set of stored variables corresponds to T1, the second to T2     
      IF(TIMEZERO) THEN
!       STORI(NSTORI) = 0
        STORI(NSTORI +3*Dim +1) = 1
        DO k1 = 1,Dim
!         STORI(NSTORI +       k1) = 0
!         STORI(NSTORI +  Dim +k1) = 0
          STORI(NSTORI +2*Dim +k1) = 2
        ENDDO
      ENDIF
     
      ! Retrieves stored information
      Nold   = STORI(NSTORI)
      BlkOld = STORI(NSTORI + 3*Dim +1)
      DO k1 = 1,Dim
        T1(k1)     = STORI(NSTORI +       k1)
        T2(k1)     = STORI(NSTORI +  Dim +k1)
        CellSt(k1) = STORI(NSTORI +2*Dim +k1)
      ENDDO
      
      ! If cells become un-blocked then set all cells as bypassed. Also re-set the old number of cells
      IF ((Blk .EQ. 0) .AND. (BlkOld .EQ. 1)) THEN
        Nold   = 0
        T1     = 1
        T2     = 0
        CellSt = 0
      ENDIF
      
      ! if the valve arm is at the top of the pole, then the number of cells to be inserted is Dim - Ncells
      ! if the valve arm is at the bottom of the pole, then the number of cells to be inserted is Ncells
      ! Note: The top arm controls or generates the negative part of the AC voltage
      ! Conversely, the bottom arm controls of generates the positive part of the AC voltage
      Nnew = Ncells
      STORI(NSTORI) = Nnew    
          
      IF (Blk .EQ. 0) THEN
        ! Note: if Nnew=Nold it means that there is no change on number of levels needed, therefore do nothing
        ! if there has been an increment in the number of levels needed
        Nold   = 0
        T1     = 1
        T2     = 0
        CellSt = 0
          
        IF (Nnew .GT. Nold) THEN
          ! Takes different acctions depending on the change of the required
          ! number of levels (if it is positive or negative):
           
          ! if current negative INSERT capacitors with highest voltage so they discharge
          IF (Iac .LT. 0.0) THEN
            k1 = 0
            DO k2 = 0,Dim-1
              n = Idx(Dim-k2)
              IF (CellSt(n) .EQ. 0) THEN
                T1(n)     = 0
                T2(n)     = 1
                CellSt(n) = 1
                k1 = k1 + 1                
              ENDIF
              IF(k1 .EQ. (Nnew-Nold)) EXIT
            ENDDO
        
          ! else--> if current positive then INSERT capacitors with lowest voltage so they charge
          ELSE
            k1 = 0           
            DO k2 = 1,Dim
              n = Idx(k2)
              IF (CellSt(n) .EQ. 0) THEN
                T1(n)     = 0
                T2(n)     = 1
                CellSt(n) = 1
                k1 = k1 + 1
              ENDIF
              IF(k1 .EQ. (Nnew-Nold)) EXIT
            ENDDO
          ENDIF
         
        ! if there has been an decrement in the number of levels needed
        ELSEIF (Nnew .LT. Nold) THEN
          ! Takes different acctions depending on the change of the required
          ! number of levels (if it is positive or negative):
          
          ! if current negative BYPASS capacitor(s) with lowest voltage
          IF (Iac .LT. 0.0) THEN
            k1 = 0
            DO k2 = 1,Dim
              n = Idx(k2)
              IF (CellSt(n) .EQ. 1) THEN
                T1(n)     = 1
                T2(n)     = 0
                CellSt(n) = 0
                k1 = k1 + 1
              ENDIF
              IF(k1 .EQ. (Nold-Nnew)) EXIT
            ENDDO
          
          
          ! else--> if current positive then BYPASS capacitor(s) with highest voltage
          ELSE  
            k1 = 0
            DO k2 = 0,Dim-1
              n = Idx(Dim-k2)
              IF (CellSt(n) .EQ. 1) THEN
                T1(n)     = 1
                T2(n)     = 0
                CellSt(n) = 0
                k1 = k1 + 1
              ENDIF
              IF(k1 .EQ. (Nold-Nnew)) EXIT
            ENDDO
          ENDIF
        ENDIF 
            
       
      ! Block all the cells in the arm
      ELSEIF (Blk .EQ. 1) THEN
        T1     = 0
        T2     = 0
        CellSt = 2
        STORI(NSTORI) = 0
        STORI(NSTORI +3*dim +1) = 1
      ENDIF
        
      
      ! Stores information for next time step
      DO k1 = 1,Dim
         STORI(NSTORI +       k1) = T1(k1)
         STORI(NSTORI +  Dim +k1) = T2(k1)
         STORI(NSTORI +2*Dim +k1) = CellSt(k1)
      ENDDO        

      NSTORI = NSTORI + 3*Dim + 2
      
      RETURN
      END SUBROUTINE MMC_Firing_HalfBridge
!
!=======================================================================
!> Determines if all the cells are in blocked state
!
