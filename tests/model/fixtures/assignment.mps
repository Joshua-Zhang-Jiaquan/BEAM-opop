* SCIP STATISTICS
*   Problem name     : assignment
*   Variables        : 9 (9 binary, 0 integer, 0 implicit integer, 0 continuous)
*   Constraints      : 6
NAME          assignment
OBJSENSE
  MIN
ROWS
 N  Obj 
 E  row_0 
 E  row_1 
 E  row_2 
 E  col_0 
 E  col_1 
 E  col_2 
COLUMNS
    INTSTART  'MARKER'                            'INTORG'                           
    x_0_0     Obj                              4  col_0                            1 
    x_0_0     row_0                            1 
    x_0_1     Obj                              2  col_1                            1 
    x_0_1     row_0                            1 
    x_0_2     row_0                            1  Obj                              8 
    x_0_2     col_2                            1 
    x_1_0     Obj                              4  row_1                            1 
    x_1_0     col_0                            1 
    x_1_1     col_1                            1  row_1                            1 
    x_1_1     Obj                              3 
    x_1_2     col_2                            1  Obj                              7 
    x_1_2     row_1                            1 
    x_2_0     row_2                            1  Obj                              3 
    x_2_0     col_0                            1 
    x_2_1     Obj                              1  row_2                            1 
    x_2_1     col_1                            1 
    x_2_2     Obj                              6  row_2                            1 
    x_2_2     col_2                            1 
    INTEND    'MARKER'                            'INTEND'                           
RHS
    RHS       row_0                            1  row_1                            1 
    RHS       row_2                            1  col_0                            1 
    RHS       col_1                            1  col_2                            1 
BOUNDS
 BV Bound     x_0_0                              
 BV Bound     x_0_1                              
 BV Bound     x_0_2                              
 BV Bound     x_1_0                              
 BV Bound     x_1_1                              
 BV Bound     x_1_2                              
 BV Bound     x_2_0                              
 BV Bound     x_2_1                              
 BV Bound     x_2_2                              
ENDATA