#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath> 
#include<cstdio>
#include "legion.h"
#include <vector>

using namespace Legion;
using namespace std;

enum Task_id
{
	Top_Level_Task,
	Testing,
	Read_Task,
	Write_Task,
};

enum FieldId{
	FID_X,
};

struct Args{
	int idx;
	Args( int _idx ) : idx(_idx){}
};


struct MultiVal{
	int node;
	int level;
	MultiVal( int _node, int _level ) : node(_node), level(_level){}
};

void top_level_task( const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime ){
	Rect<1> tree_rect(0LL, static_cast<coord_t>(pow(2, 3)));
    IndexSpace is = runtime->create_index_space(ctx, tree_rect);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(MultiVal), FID_X);
    }

    // For 1st logical region
    LogicalRegion lr1 = runtime->create_logical_region(ctx, is, fs);
    Args arg(123);
    TaskLauncher testing_launcher(Testing, TaskArgument(&arg, sizeof(Args)));
    testing_launcher.add_region_requirement(RegionRequirement(lr1, WRITE_DISCARD, EXCLUSIVE, lr1));
    testing_launcher.add_field(0, FID_X);
    runtime->execute_task(ctx, testing_launcher);
}

void testing_task( const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime ){
	Args args = task->is_index_space ? *(const Args *) task->local_args
    : *(const Args *) task->args;
    int idx = args.idx;
    cout<<"Received "<<idx<<endl;
    LogicalRegion lr = regions[0].get_logical_region();
    MultiVal ArgMultival(0,0);
    TaskLauncher write_task_launcher( Write_Task, TaskArgument(&ArgMultival, sizeof(MultiVal)) );
    write_task_launcher.add_region_requirement(RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
    write_task_launcher.add_field(0,FID_X);
    runtime->execute_task( ctx, write_task_launcher );

    TaskLauncher read_task_launcher( Read_Task, TaskArgument(&ArgMultival, sizeof(MultiVal)) );
    read_task_launcher.add_region_requirement(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
    read_task_launcher.add_field(0,FID_X);
    runtime->execute_task( ctx , read_task_launcher );
}





void write_task( const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime ){
	    const FieldAccessor<WRITE_DISCARD, MultiVal, 1> write_acc(regions[0], FID_X);
	    for( int i = 0 ; i < 5 ; i++ ){
	    	MultiVal temp(i,i+1);
	    	write_acc[i]= temp;
	    }
}


void read_task( const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime ){
	    const FieldAccessor<READ_ONLY, MultiVal, 1> read_acc(regions[0], FID_X);
	    for( int i = 0 ; i < 8 ; i++ ){
	    	cout<<read_acc[i].node<<"~"<<read_acc[i].level<<endl;
	    }
}


int main(int argc, char** argv){
		
	Runtime::set_top_level_task_id(Top_Level_Task);

	{
		TaskVariantRegistrar registrar(Top_Level_Task, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
	}

	{
		TaskVariantRegistrar registrar(Testing, "refine_inter");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<testing_task>(registrar, "refine_inter");
	}

	{
		TaskVariantRegistrar registrar(Read_Task, "read_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<read_task>(registrar, "read_task");
	}

	{
		TaskVariantRegistrar registrar(Write_Task, "write_Task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<write_task>(registrar, "write_task");
	}


	return Runtime::start(argc,argv);

}

