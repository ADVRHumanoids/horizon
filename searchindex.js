Search.setIndex({docnames:["horizon","horizon.ros","horizon.solvers","horizon.transcriptions","horizon.utils","index","modules"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.todo":1,sphinx:55},filenames:["horizon.rst","horizon.ros.rst","horizon.solvers.rst","horizon.transcriptions.rst","horizon.utils.rst","index.rst","modules.rst"],objects:{"":{horizon:[0,0,0,"-"]},"horizon.functions":{Constraint:[0,1,1,""],CostFunction:[0,1,1,""],Function:[0,1,1,""],FunctionsContainer:[0,1,1,""],ResidualFunction:[0,1,1,""]},"horizon.functions.Constraint":{getBounds:[0,2,1,""],getLowerBounds:[0,2,1,""],getType:[0,2,1,""],getUpperBounds:[0,2,1,""],setBounds:[0,2,1,""],setLowerBounds:[0,2,1,""],setNodes:[0,2,1,""],setUpperBounds:[0,2,1,""]},"horizon.functions.CostFunction":{getType:[0,2,1,""]},"horizon.functions.Function":{deserialize:[0,2,1,""],getDim:[0,2,1,""],getFunction:[0,2,1,""],getImpl:[0,2,1,""],getName:[0,2,1,""],getNodes:[0,2,1,""],getParameters:[0,2,1,""],getType:[0,2,1,""],getVariables:[0,2,1,""],serialize:[0,2,1,""],setNodes:[0,2,1,""]},"horizon.functions.FunctionsContainer":{addFunction:[0,2,1,""],build:[0,2,1,""],deserialize:[0,2,1,""],getCnstr:[0,2,1,""],getCnstrDim:[0,2,1,""],getCost:[0,2,1,""],getFunction:[0,2,1,""],removeFunction:[0,2,1,""],serialize:[0,2,1,""],setNNodes:[0,2,1,""]},"horizon.functions.ResidualFunction":{getType:[0,2,1,""]},"horizon.misc_function":{checkNodes:[0,3,1,""],checkValueEntry:[0,3,1,""],listOfListFLOATtoINT:[0,3,1,""],unravelElements:[0,3,1,""]},"horizon.nodes":{Nodes:[0,1,1,""]},"horizon.nodes.Nodes":{getValue:[0,2,1,""]},"horizon.problem":{Problem:[0,1,1,""],pickleable:[0,3,1,""]},"horizon.problem.Problem":{createConstraint:[0,2,1,""],createCost:[0,2,1,""],createFinalConstraint:[0,2,1,""],createFinalCost:[0,2,1,""],createFinalResidual:[0,2,1,""],createInputVariable:[0,2,1,""],createIntermediateConstraint:[0,2,1,""],createIntermediateCost:[0,2,1,""],createIntermediateResidual:[0,2,1,""],createParameter:[0,2,1,""],createResidual:[0,2,1,""],createSingleParameter:[0,2,1,""],createSingleVariable:[0,2,1,""],createStateVariable:[0,2,1,""],createVariable:[0,2,1,""],deserialize:[0,2,1,""],evalFun:[0,2,1,""],getConstraints:[0,2,1,""],getCosts:[0,2,1,""],getDt:[0,2,1,""],getDynamics:[0,2,1,""],getInitialState:[0,2,1,""],getInput:[0,2,1,""],getNNodes:[0,2,1,""],getParameters:[0,2,1,""],getState:[0,2,1,""],getVariables:[0,2,1,""],removeConstraint:[0,2,1,""],removeCostFunction:[0,2,1,""],removeVariable:[0,2,1,""],resetDynamics:[0,2,1,""],save:[0,2,1,""],scopeNodeConstraints:[0,2,1,""],scopeNodeCostFunctions:[0,2,1,""],scopeNodeVars:[0,2,1,""],serialize:[0,2,1,""],setDt:[0,2,1,""],setDynamics:[0,2,1,""],setInitialState:[0,2,1,""],setNNodes:[0,2,1,""],toParameter:[0,2,1,""]},"horizon.ros":{replay_trajectory:[1,0,0,"-"],tf_broadcaster_simple:[1,0,0,"-"],utils:[1,0,0,"-"]},"horizon.ros.replay_trajectory":{normalize_quaternion:[1,3,1,""],replay_trajectory:[1,1,1,""]},"horizon.ros.replay_trajectory.replay_trajectory":{publishContactForces:[1,2,1,""],publish_joints:[1,2,1,""],replay:[1,2,1,""],setSlowDownFactor:[1,2,1,""],sleep:[1,2,1,""]},"horizon.ros.tf_broadcaster_simple":{TransformBroadcaster:[1,1,1,""]},"horizon.ros.tf_broadcaster_simple.TransformBroadcaster":{sendTransform:[1,2,1,""]},"horizon.ros.utils":{roslaunch:[1,3,1,""]},"horizon.solvers":{blocksqp:[2,0,0,"-"],ilqr:[2,0,0,"-"],ipopt:[2,0,0,"-"],nlpsol:[2,0,0,"-"],solver:[2,0,0,"-"],sqp:[2,0,0,"-"]},"horizon.solvers.blocksqp":{BlockSqpSolver:[2,1,1,""]},"horizon.solvers.blocksqp.BlockSqpSolver":{configure_rti:[2,2,1,""]},"horizon.solvers.ilqr":{SolverILQR:[2,1,1,""]},"horizon.solvers.ilqr.SolverILQR":{configure_rti:[2,2,1,""],getDt:[2,2,1,""],getSolutionDict:[2,2,1,""],print_timings:[2,2,1,""],save:[2,2,1,""],set_iteration_callback:[2,2,1,""],solve:[2,2,1,""]},"horizon.solvers.ipopt":{IpoptSolver:[2,1,1,""]},"horizon.solvers.nlpsol":{NlpsolSolver:[2,1,1,""]},"horizon.solvers.nlpsol.NlpsolSolver":{build:[2,2,1,""],getConstraintSolutionDict:[2,2,1,""],getDt:[2,2,1,""],getSolutionDict:[2,2,1,""],solve:[2,2,1,""]},"horizon.solvers.solver":{Solver:[2,1,1,""]},"horizon.solvers.solver.Solver":{configure_rti:[2,2,1,""],getDt:[2,2,1,""],getSolutionDict:[2,2,1,""],make_solver:[2,4,1,""],solve:[2,2,1,""]},"horizon.solvers.sqp":{GNSQPSolver:[2,1,1,""]},"horizon.solvers.sqp.GNSQPSolver":{getAlpha:[2,2,1,""],getBeta:[2,2,1,""],getConstraintNormIterations:[2,2,1,""],getConstraintSolutionDict:[2,2,1,""],getDt:[2,2,1,""],getHessianComputationTime:[2,2,1,""],getLineSearchComputationTime:[2,2,1,""],getObjectiveIterations:[2,2,1,""],getQPComputationTime:[2,2,1,""],getSolutionDict:[2,2,1,""],setAlphaMin:[2,2,1,""],setBeta:[2,2,1,""],set_iteration_callback:[2,2,1,""],solve:[2,2,1,""]},"horizon.transcriptions":{integrators:[3,0,0,"-"],methods:[3,0,0,"-"],transcriptor:[3,0,0,"-"],trial_integrator:[3,0,0,"-"]},"horizon.transcriptions.integrators":{EULER:[3,3,1,""],LEAPFROG:[3,3,1,""],RK2:[3,3,1,""],RK4:[3,3,1,""]},"horizon.transcriptions.methods":{DirectCollocation:[3,1,1,""],MultipleShooting:[3,1,1,""]},"horizon.transcriptions.methods.MultipleShooting":{setDefaultIntegrator:[3,2,1,""]},"horizon.transcriptions.transcriptor":{Transcriptor:[3,1,1,""]},"horizon.transcriptions.transcriptor.Transcriptor":{make_method:[3,4,1,""]},"horizon.transcriptions.trial_integrator":{RK4:[3,3,1,""]},"horizon.type_doc":{BoundsDict:[0,1,1,""]},"horizon.utils":{collision:[4,0,0,"-"],kin_dyn:[4,0,0,"-"],mat_storer:[4,0,0,"-"],plotter:[4,0,0,"-"],refiner:[4,0,0,"-"],resampler_trajectory:[4,0,0,"-"],rti:[4,0,0,"-"],utils:[4,0,0,"-"]},"horizon.utils.collision":{CollisionHandler:[4,1,1,""]},"horizon.utils.collision.CollisionHandler":{clamp:[4,4,1,""],collision_to_capsule:[4,4,1,""],compute_distances:[4,2,1,""],dist_capsule_capsule:[4,4,1,""],dist_segment_segment:[4,4,1,""],get_function:[4,2,1,""]},"horizon.utils.kin_dyn":{ForwardDynamics:[4,1,1,""],InverseDynamics:[4,1,1,""],InverseDynamicsMap:[4,1,1,""],linearized_friction_cone:[4,3,1,""],linearized_friction_cone_map:[4,3,1,""],surface_point_contact:[4,3,1,""]},"horizon.utils.kin_dyn.ForwardDynamics":{call:[4,2,1,""]},"horizon.utils.kin_dyn.InverseDynamics":{call:[4,2,1,""]},"horizon.utils.kin_dyn.InverseDynamicsMap":{call:[4,2,1,""]},"horizon.utils.mat_storer":{matStorer:[4,1,1,""],matStorerIO:[4,1,1,""],setInitialGuess:[4,3,1,""]},"horizon.utils.mat_storer.matStorer":{append:[4,2,1,""],load:[4,2,1,""],save:[4,2,1,""],store:[4,2,1,""]},"horizon.utils.mat_storer.matStorerIO":{append:[4,2,1,""],argParse:[4,2,1,""],load:[4,2,1,""],save:[4,2,1,""],store:[4,2,1,""]},"horizon.utils.plotter":{PlotterHorizon:[4,1,1,""]},"horizon.utils.plotter.PlotterHorizon":{plotFunction:[4,2,1,""],plotFunctions:[4,2,1,""],plotVariable:[4,2,1,""],plotVariables:[4,2,1,""],setSolution:[4,2,1,""]},"horizon.utils.refiner":{Refiner:[4,1,1,""]},"horizon.utils.refiner.Refiner":{addProximalCosts:[4,2,1,""],expandDt:[4,2,1,""],expand_nodes:[4,2,1,""],find_nodes_to_inject:[4,2,1,""],getAugmentedProblem:[4,2,1,""],getSolution:[4,2,1,""],get_node_time:[4,2,1,""],group_elements:[4,2,1,""],resetFunctions:[4,2,1,""],resetInitialGuess:[4,2,1,""],resetProblem:[4,2,1,""],resetVarBounds:[4,2,1,""],solveProblem:[4,2,1,""]},"horizon.utils.resampler_trajectory":{resample_input:[4,3,1,""],resample_torques:[4,3,1,""],resampler:[4,3,1,""],second_order_resample_integrator:[4,3,1,""]},"horizon.utils.rti":{RealTimeIteration:[4,1,1,""]},"horizon.utils.rti.RealTimeIteration":{integrate:[4,2,1,""],run:[4,2,1,""]},"horizon.utils.utils":{double_integrator:[4,3,1,""],double_integrator_with_floating_base:[4,3,1,""],jac:[4,3,1,""],quaterion_product:[4,3,1,""],skew:[4,3,1,""],toRot:[4,3,1,""]},"horizon.variables":{AbstractAggregate:[0,1,1,""],AbstractVariable:[0,1,1,""],AbstractVariableView:[0,1,1,""],Aggregate:[0,1,1,""],InputAggregate:[0,1,1,""],InputVariable:[0,1,1,""],OffsetAggregate:[0,1,1,""],OffsetParameter:[0,1,1,""],OffsetVariable:[0,1,1,""],Parameter:[0,1,1,""],ParameterView:[0,1,1,""],SingleParameter:[0,1,1,""],SingleParameterView:[0,1,1,""],SingleVariable:[0,1,1,""],SingleVariableView:[0,1,1,""],StateAggregate:[0,1,1,""],StateVariable:[0,1,1,""],Variable:[0,1,1,""],VariableView:[0,1,1,""],VariablesContainer:[0,1,1,""]},"horizon.variables.AbstractAggregate":{getVars:[0,2,1,""]},"horizon.variables.AbstractVariable":{getDim:[0,2,1,""],getName:[0,2,1,""],getOffset:[0,2,1,""]},"horizon.variables.AbstractVariableView":{getName:[0,2,1,""]},"horizon.variables.Aggregate":{addVariable:[0,2,1,""],getBounds:[0,2,1,""],getInitialGuess:[0,2,1,""],getLowerBounds:[0,2,1,""],getUpperBounds:[0,2,1,""],getVarIndex:[0,2,1,""],getVarOffset:[0,2,1,""],removeVariable:[0,2,1,""],setBounds:[0,2,1,""],setInitialGuess:[0,2,1,""],setLowerBounds:[0,2,1,""],setUpperBounds:[0,2,1,""]},"horizon.variables.OffsetAggregate":{getVarIndex:[0,2,1,""]},"horizon.variables.OffsetParameter":{getImpl:[0,2,1,""],getName:[0,2,1,""],getNodes:[0,2,1,""]},"horizon.variables.OffsetVariable":{getImpl:[0,2,1,""],getName:[0,2,1,""],getNodes:[0,2,1,""]},"horizon.variables.Parameter":{assign:[0,2,1,""],getImpl:[0,2,1,""],getName:[0,2,1,""],getNodes:[0,2,1,""],getParOffset:[0,2,1,""],getParOffsetDict:[0,2,1,""],getValues:[0,2,1,""]},"horizon.variables.ParameterView":{assign:[0,2,1,""],getValues:[0,2,1,""]},"horizon.variables.SingleParameter":{assign:[0,2,1,""],getImpl:[0,2,1,""],getName:[0,2,1,""],getNodes:[0,2,1,""],getParOffset:[0,2,1,""],getParOffsetDict:[0,2,1,""],getValues:[0,2,1,""]},"horizon.variables.SingleParameterView":{assign:[0,2,1,""]},"horizon.variables.SingleVariable":{getBounds:[0,2,1,""],getImpl:[0,2,1,""],getImplDim:[0,2,1,""],getInitialGuess:[0,2,1,""],getLowerBounds:[0,2,1,""],getName:[0,2,1,""],getNodes:[0,2,1,""],getUpperBounds:[0,2,1,""],getVarOffset:[0,2,1,""],getVarOffsetDict:[0,2,1,""],setBounds:[0,2,1,""],setInitialGuess:[0,2,1,""],setLowerBounds:[0,2,1,""],setUpperBounds:[0,2,1,""]},"horizon.variables.SingleVariableView":{setBounds:[0,2,1,""],setInitialGuess:[0,2,1,""],setLowerBounds:[0,2,1,""],setUpperBounds:[0,2,1,""]},"horizon.variables.Variable":{getBounds:[0,2,1,""],getImpl:[0,2,1,""],getImplDim:[0,2,1,""],getInitialGuess:[0,2,1,""],getLowerBounds:[0,2,1,""],getName:[0,2,1,""],getNodes:[0,2,1,""],getUpperBounds:[0,2,1,""],getVarOffset:[0,2,1,""],getVarOffsetDict:[0,2,1,""],setBounds:[0,2,1,""],setInitialGuess:[0,2,1,""],setLowerBounds:[0,2,1,""],setUpperBounds:[0,2,1,""]},"horizon.variables.VariableView":{setBounds:[0,2,1,""],setInitialGuess:[0,2,1,""],setLowerBounds:[0,2,1,""],setUpperBounds:[0,2,1,""]},"horizon.variables.VariablesContainer":{createVar:[0,2,1,""],deserialize:[0,2,1,""],getInputVars:[0,2,1,""],getPar:[0,2,1,""],getParList:[0,2,1,""],getStateVars:[0,2,1,""],getVar:[0,2,1,""],getVarList:[0,2,1,""],removeVar:[0,2,1,""],serialize:[0,2,1,""],setInputVar:[0,2,1,""],setNNodes:[0,2,1,""],setParameter:[0,2,1,""],setSingleParameter:[0,2,1,""],setSingleVar:[0,2,1,""],setStateVar:[0,2,1,""],setVar:[0,2,1,""]},horizon:{functions:[0,0,0,"-"],misc_function:[0,0,0,"-"],nodes:[0,0,0,"-"],problem:[0,0,0,"-"],ros:[1,0,0,"-"],solvers:[2,0,0,"-"],transcriptions:[3,0,0,"-"],type_doc:[0,0,0,"-"],utils:[4,0,0,"-"],variables:[0,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","classmethod","Python class method"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:classmethod"},terms:{"0x7f113af1a400":0,"0x7f113af6f340":0,"0x7f113b0e99d0":0,"0x7f113c4c89a0":0,"0x7f113c4c8a00":0,"2nd":3,"4th":3,"abstract":[0,2],"case":[],"class":[0,1,2,3,4],"default":[0,2,3,4],"final":[0,3],"float":[2,3,4],"function":[3,4,6],"import":[0,3],"int":[0,3],"new":[0,4],"return":[0,2,3,4],"true":[0,1,4],"var":0,"while":4,FOR:0,The:[0,2,3],Used:0,_mockobject:[0,3],a11:[],a13:[],a21:[],a32:[],a_r:4,abc:[0,2,3],abstr:0,abstractaggreg:0,abstractvari:0,abstractvariableview:0,acceler:4,accord:4,account:3,acquaint:5,act:4,action:2,activ:0,active_nod:0,actual:3,add:[0,3,4],added:0,addfunct:0,addproximalcost:4,addvari:0,after:0,afterward:0,aggreg:0,aim:2,algorithm:5,all:[0,3],allow:0,along:[0,3],alpha_min:2,also:[0,3],alwai:0,angular:4,ani:5,append:4,appli:0,appropri:2,approxim:3,arg:0,argpars:4,arrai:[0,2,4],art:5,assign:0,associ:4,autodoc:[0,3],automat:0,auxiliari:3,b21:[],b32:[],base:[0,1,2,3,4,5],base_link:4,befor:0,belong:0,beta:2,between:[1,3,4],blocksqp:[0,6],blocksqpsolv:2,bool:[0,2],bound:0,boundsdict:0,brief:5,browser:5,build:[0,2],built:[0,4],call:[2,4],can:[0,3,5],capsule_1:4,capsule_2:4,cart_pol:6,cart_pole_fd:[],cart_pole_fd_ilqr:6,cart_pole_feedback:6,cart_pole_feedback_solv:[],cart_pole_feedback_sqp:[],cart_pole_final_tim:[],cart_pole_manu:6,cart_pole_sin:[],casadi:[0,4,5],casadi_horizon:5,casadi_kin_dyn:4,casadi_typ:3,check:4,checknod:0,checkvalueentri:0,child_frame_id:1,choos:0,chrome:5,clamp:4,classmethod:[2,3,4],clone:5,cmth:3,code:0,collect:0,collis:[0,6],collision_exampl:6,collision_test:6,collision_to_capsul:4,collisionhandl:4,collisiontest:[],colloc:3,compon:4,compphi:3,comput:4,compute_dist:4,compute_hessian:6,concret:2,cone:4,configure_rti:2,consid:[0,4],constant:4,constrain:[3,4],constraint:[0,3,4],construct:[2,3,4],contact:4,contact_fram:4,contain:[0,2,3,4],content:[5,6],control:[0,3,4,5],convert:3,coordin:4,copy_external_depend:6,cost:0,costfunct:0,could:3,crash_if_suboptim:0,creat:[0,4],createconstraint:0,createcost:0,createfinalconstraint:0,createfinalcost:0,createfinalresidu:0,createinputvari:0,createintermediateconstraint:0,createintermediatecost:0,createintermediateresidu:0,createparamet:0,createresidu:0,createsingleparamet:0,createsinglevari:0,createstatevari:0,createvar:0,createvari:0,dae:[3,4],decis:[0,3],deerial:0,defect:3,defin:[0,3,4],definit:0,degre:3,depend:[2,3],deriv:[0,3,4],describ:0,descript:[0,4],deseri:0,desir:[0,3,5],desired_dt:4,dict:[0,2,3,4],dict_valu:4,dictionari:[0,2,3,4],differ:[3,4],differenti:5,dim:[0,4],dimens:0,dimension:3,direct:[3,5],directcolloc:3,discret:[2,3],displai:5,dist_capsule_capsul:4,dist_segment_seg:4,divid:0,done:4,double_integr:4,double_integrator_with_floating_bas:4,down:1,dt_rk:3,dummy_nod:0,dure:4,dynam:[0,4],each:[0,2],element:0,els:5,empti:0,equalityconstrain:[],eras:0,euler:3,evalfun:0,evalu:[0,4],evolut:3,exampl:[0,4,6],except:0,exercis:[],expand_nod:4,expanddt:4,experiment_aug:[],express:4,ext:[0,3],extra:4,f_rk:3,factor:1,factori:2,fals:[0,4],fator:1,ff_r_cf:4,file:1,file_nam:4,fill:2,find:2,find_nodes_to_inject:4,firefox:5,fixtur:[],flag:2,focu:5,folder:5,follow:[0,4],forc:4,force_reference_fram:[1,4],forward:4,forwarddynam:4,found:0,frame:4,frame_force_map:[1,4],frame_id:1,frame_res_force_map:4,framework:0,friciton:4,from:[0,2,4],fun:0,fun_nam:0,function_string_list:4,functionscontain:0,gather:4,gener:[0,2,4],get:[0,2],get_funct:4,get_node_tim:4,getalpha:2,getaugmentedproblem:4,getbeta:2,getbound:0,getcnstr:0,getcnstrdim:0,getconstraint:0,getconstraintnormiter:2,getconstraintsolutiondict:2,getcost:0,getdim:0,getdt:[0,2],getdynam:0,getfunct:0,gethessiancomputationtim:2,getimpl:0,getimpldim:0,getinitialguess:0,getinitialst:0,getinput:0,getinputvar:0,getlinesearchcomputationtim:2,getlowerbound:0,getnam:0,getnnod:0,getnod:0,getobjectiveiter:2,getoffset:0,getpar:0,getparamet:0,getparlist:0,getparoffset:0,getparoffsetdict:0,getqpcomputationtim:2,getsolut:4,getsolutiondict:2,getstat:0,getstatevar:0,getter:0,gettyp:0,getupperbound:0,getvalu:0,getvar:0,getvari:0,getvarindex:0,getvarlist:0,getvaroffset:0,getvaroffsetdict:0,github:5,given:[0,2,3,4],global:4,gnsqpsolver:2,greatli:0,grid:4,group_el:4,guess:0,hessian:4,hook:[],horizon:6,html:3,http:3,iii:2,ilqr:[0,6],ilqr_test:6,implement:[0,2,3,4],independ:0,index:5,indic:[0,2],infinit:3,initi:0,input:[0,3,4],input_r:4,input_vec:4,inputaggreg:0,inputvari:0,insert:0,insid:0,instanc:[0,2,3],integr:[0,4,6],integrator_typ:3,interfac:2,intern:[0,4],interv:0,intuit:5,invers:4,inversedynam:4,inversedynamicsmap:4,ipopt:[0,6],ipoptsolv:2,ipynb:5,is_floating_bas:1,iter:0,its:[0,2,3],itself:0,jac:4,jac_test:4,jacobian:4,joint:4,joint_list:1,jupyt:5,kei:2,kept:0,kin_dyn:[0,6],kindyn:[1,4],kutta:3,last:0,launch:1,leapfrog:3,lectur:3,legend:4,librari:5,lies:4,linear:[0,4],linearized_friction_con:4,linearized_friction_cone_map:4,link:4,list:[0,4],listoflist:0,listoflistfloattoint:0,load:[0,4],local:[1,4],local_world_align:4,locat:1,logger:[0,3,4],logging_level:0,lower:0,lsole:4,lwr_ik:6,mackinnon:3,main:0,make:2,make_method:3,make_problem:[],make_solv:2,manag:0,map:4,marker:4,mat_stor:[0,6],matrix:[0,4],matstor:4,matstorerio:4,mechan:0,method:[0,5,6],methodnam:[],misc_funct:6,model:4,modul:[5,6],more:0,multipl:3,multipleshoot:3,multipli:0,must:0,myvar:0,n_node:0,name:[0,2,4],natur:0,nddot:4,ndot:4,need:2,new_nodes_vec:4,nlp:5,nlpsol:[0,6],nlpsolsolv:2,nnone:2,node11:3,node4:3,node:[2,4,6],node_tim:4,nodes_dt:4,nodes_self:0,non:0,none:[0,1,2,3,4],nonlinear:5,normalize_quaternion:1,note:[0,3,4],notebook:5,noth:5,number:[0,3,4],numpi:[0,4],obj:0,object:[0,1,3,4],ode:[3,4],off:0,offset:0,offset_vari:0,offsetaggreg:0,offsetparamet:0,offsetvari:0,omega:4,one:0,onli:[0,4],open:5,opt:[2,3,4],opti_spot:6,optim:[0,2,3,5],option:[0,2,3],order:[0,3],ordereddict:0,orient:4,origin:4,over:[0,3],overrid:0,p_0:0,p_1:0,p_fp1:4,p_fp2:4,p_fq1:4,p_fq2:4,p_n:0,p_re:4,packag:[5,6],page:5,pair:0,par_impl:0,param:[0,1,3,4],paramet:[0,2,3,4],parameterview:0,parametr:0,paramter_support_test:6,paramtest:[],parent:0,parent_nam:0,part:4,peopl:3,perform:3,period:4,pickleabl:0,pip:5,plan:0,plane:4,plane_dict:4,playground:6,plotfunct:4,plotter:[0,6],plotterhorizon:4,plotvari:4,point:[0,4],pointer:3,polynomi:3,portion:0,pos:1,posit:4,power:5,prb:[2,3,4],prepar:5,previou:[0,4],print_tim:2,prob:3,problem:[2,3,4,5,6],process:3,product:4,project:[0,2],publish_joint:1,publishcontactforc:1,python:5,q_replai:1,q_rk:3,qddot:4,qddotj:4,qdot:4,qdotj:4,qp_solver_plugin:2,quad:[3,4],quadratur:[3,4],quadruped_jump:6,quadruped_jump_fd:6,quadrupedal_walk:6,quat:4,quatdot:4,quaterion_product:4,quaternion:4,rather:[0,3],real:3,realtimeiter:4,reduc:5,refer:[0,4],referencefram:[1,4],refin:[0,6],rel:0,reli:5,remov:0,removeconstraint:0,removecostfunct:0,removefunct:0,removevar:0,removevari:0,replai:1,replay:[0,6],replay_trajectori:[0,6],replayer_fd:[0,6],replayer_fd_mx:[0,6],replayer_mx:[0,6],repo:5,repres:4,represent:0,requir:2,resampl:4,resample_input:4,resample_torqu:4,resampler_trajectori:[0,6],resetdynam:0,resetfunct:4,resetinitialguess:4,resetproblem:4,resetvarbound:4,residu:0,residualfunct:0,result:0,retriev:[0,2],rk2:3,rk4:[3,4],robot:5,rope_jump:6,roped_robot:6,roped_robot_rappel:6,roped_robot_sim:6,ros:[0,6],roslaunch:1,rot:1,rotat:4,rti:[0,2,6],run:[1,4,5],rung:3,runtest:[],rviz_jump:6,same:0,sampl:4,save:[0,2,4],scheme:[2,3],scope:0,scopenodeconstraint:0,scopenodecostfunct:0,scopenodevar:0,search:5,sec:1,second:1,second_order_resample_integr:4,seen:3,segment:3,self:0,send_to_gazebo:[0,6],sendtransform:1,separ:1,sequenc:1,serial:0,set:[0,1],set_iteration_callback:2,setalphamin:2,setbeta:2,setbound:0,setdefaultintegr:3,setdt:0,setdynam:0,setinitialguess:[0,4],setinitialst:0,setinputvar:0,setlowerbound:0,setnnod:0,setnod:0,setparamet:0,setsingleparamet:0,setsinglevar:0,setslowdownfactor:1,setsolut:4,setstatevar:0,setter:0,setup:[5,6],setupperbound:0,setvar:0,shift:0,shoot:[0,3],should:5,show_bound:4,simplifi:0,sinc:[0,3],singl:[0,2],singleparamet:0,singleparameterview:0,singlevari:0,singlevariableview:0,skew:4,sleep:1,slow:1,slow_down_factor:1,solut:[0,2,3,4],solv:[0,2,3,5],solveproblem:4,solver:[0,3,4,5,6],solver_lq_test:6,solver_lq_test_gnssqp:6,solver_plugin:2,solver_typ:[],solverconsist:[],solverilqr:2,someth:6,space:4,special:5,specifi:0,sphinx:[0,3],spot:6,spot_backflip_manu:[],spot_direct_collo:[],spot_fd:[],spot_fd_manu:[],spot_fd_step_manu:[],spot_ilqr_bal:6,spot_ilqr_front_jump:6,spot_ilqr_jump:[],spot_jump:[],spot_jump_fb:[],spot_jump_forward:[],spot_jump_re_refin:[],spot_jump_refined_glob:[],spot_jump_refined_loc:[],spot_jump_simplifi:[],spot_jump_twist:[],spot_jump_twist_refin:[],spot_leap:[],spot_leg_rais:[],spot_mot:6,spot_on_box:[],spot_step:[],spot_step_fd:[],spot_step_manu:[],spot_walk:6,spot_wheeli:[],spot_wheelie_manu:[],sqp:[0,6],sqp_problem:6,state:[0,3,4,5],state_var_impl:2,state_vec:4,stateaggreg:0,stateread:4,statevari:0,step:[2,3],store:[0,4],str:[0,2,3],strategi:3,string:[0,2],structur:0,submodul:6,subpackag:6,success:2,suitabl:2,summari:[0,4],support:5,surfac:4,surface_point_contact:4,swing:6,symbol:0,symmetr:4,system:[0,5],t_co:0,tag:0,taht:5,take:[2,3],tau:4,tau_ext:4,tau_r:4,techniqu:3,term:[3,4],test:[4,6],test_blocksqp_vs_ilqr:[],test_blocksqp_vs_ipopt:[],test_bound:[],test_bounds_2:[],test_bounds_input:[],test_boundsarrai:[],test_constr_bound:[],test_constraintbound:[],test_get_set:6,test_gnsqp:[],test_horizon:6,test_initial_guess:[],test_intermediateconstraint:[],test_parallel_capsul:[],test_param_valu:[],test_paramet:[],test_prev:[],test_robot:[],test_simple_constrain:[],test_singleparamet:[],test_singlevari:[],test_touching_capsul:[],test_touching_capsules_rot:[],test_vari:[],test_view:[],testcas:[],tf_broadcaster_simpl:[0,6],than:3,them:0,thi:[0,3,4],thread:1,throughout:0,time:[1,3,4],todo:4,tool:[0,5],toparamet:0,torot:4,torqu:4,total:0,trajectori:[0,1,3],transcript:[0,6],transcriptor:[0,6],transformbroadcast:1,trial_hessian:6,trial_integr:[0,6],trial_single_var:6,try_variable_convers:6,tutori:5,two:4,type:[0,2,3,4],type_doc:6,typic:2,u_opt:[2,4],u_r:4,u_rk:3,union:0,unit_mass:6,unit_mass_colloc:6,unittest:[],unravelel:0,until:5,updat:0,upper:0,urdf:4,urdf_parser_pi:4,urdfstr:4,usag:[2,4],used:[0,3,4],used_par:0,used_var:0,useful:0,useless:0,user:0,using:[0,4,5],util:[0,6],v_re:4,val:[0,2],valu:[0,2],var_impl:0,var_nam:0,var_slic:0,var_string_list:4,var_typ:0,variabl:[2,3,4,6],variablescontain:0,variableview:0,vec:4,vec_to_expand:4,vector:[0,2,4],veloc:4,vis_refiner_glob:[0,6],vis_refiner_loc:[0,6],want:5,warn:0,wdoti:4,wdotx:4,wdotz:4,web:5,well:4,where:[0,1,4],which:[0,3,4],work:0,world:4,www:3,x0_rk:3,x_0:0,x_1:0,x_n:0,x_opt:2,x_rk:3,xdot:[0,3,4],xmax:4,xmin:4,yet:5,yield:3,you:5,your:5,zero:0},titles:["horizon package","horizon.ros package","horizon.solvers package","horizon.transcriptions package","horizon.utils package","Welcome to horizon\u2019s documentation!","casadi_horizon"],titleterms:{"function":0,blocksqp:2,cart_pol:[],cart_pole_fd:[],cart_pole_fd_ilqr:[],cart_pole_feedback:[],cart_pole_feedback_solv:[],cart_pole_feedback_sqp:[],cart_pole_final_tim:[],cart_pole_manu:[],cart_pole_sin:[],casadi_horizon:6,collis:4,collision_exampl:[],collision_test:[],compute_hessian:[],content:[0,1,2,3,4],copy_external_depend:[],document:5,exampl:[],experiment_aug:[],featur:5,get:5,horizon:[0,1,2,3,4,5],ilqr:2,ilqr_test:[],indic:5,instal:5,integr:3,ipopt:2,kin_dyn:4,lwr_ik:[],mat_stor:4,method:3,misc_funct:0,modul:[0,1,2,3,4],nlpsol:2,node:0,opti_spot:[],packag:[0,1,2,3,4],paramter_support_test:[],playground:[],plotter:4,problem:0,quadruped_jump:[],quadruped_jump_fd:[],quadrupedal_walk:[],refin:4,replay:4,replay_trajectori:1,replayer_fd:4,replayer_fd_mx:4,replayer_mx:4,resampler_trajectori:4,rope_jump:[],roped_robot:[],roped_robot_rappel:[],roped_robot_sim:[],ros:1,rti:4,rviz_jump:[],send_to_gazebo:4,setup:[],solver:2,solver_lq_test:[],solver_lq_test_gnssqp:[],someth:[],spot:[],spot_backflip_manu:[],spot_direct_collo:[],spot_fd:[],spot_fd_manu:[],spot_fd_step_manu:[],spot_ilqr_bal:[],spot_ilqr_front_jump:[],spot_ilqr_jump:[],spot_jump:[],spot_jump_fb:[],spot_jump_forward:[],spot_jump_re_refin:[],spot_jump_refined_glob:[],spot_jump_refined_loc:[],spot_jump_simplifi:[],spot_jump_twist:[],spot_jump_twist_refin:[],spot_leap:[],spot_leg_rais:[],spot_mot:[],spot_on_box:[],spot_step:[],spot_step_fd:[],spot_step_manu:[],spot_walk:[],spot_wheeli:[],spot_wheelie_manu:[],sqp:2,sqp_problem:[],start:5,submodul:[0,1,2,3,4],subpackag:0,swing:[],tabl:5,test:[],test_get_set:[],test_horizon:[],tf_broadcaster_simpl:1,transcript:3,transcriptor:3,trial_hessian:[],trial_integr:3,trial_single_var:[],try_variable_convers:[],type_doc:0,unit_mass:[],unit_mass_colloc:[],util:[1,4],variabl:0,vis_refiner_glob:4,vis_refiner_loc:4,welcom:5}})