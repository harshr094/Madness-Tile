#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath> 
#include <cstdio>
#include "legion.h"
#include <vector>
#include <queue>
#include <utility>

using namespace Legion;
using namespace std;

enum TASK_IDs{
    TOP_LEVEL_TASK_ID,
    REFINE_INTER_TASK_ID,
    REFINE_INTRA_TASK_ID,
    PRINT_TASK_ID,
    COMPRESS_INTRA_TASK_ID,
    COMPRESS_INTER_TASK_ID,
};

enum FieldId{
    FID_X,
};

struct Arguments {
    /* level of the node in the binary tree. Root is at level 0 */
    int n;

    /* labeling of the node in the binary tree. Root has the value label = 0 
    * Node with (n, l) has it's left child at (n + 1, 2 * l) and it's right child at (n + 1, 2 * l + 1)
    */
    int l;

    int max_depth;

    coord_t idx;

    long int gen;

    Color partition_color;

    int actual_max_depth;

    int tile_height;

    Arguments(int _n, int _l, int _max_depth, coord_t _idx, Color _partition_color, int _actual_max_depth=0, int _tile_height=1 )
        : n(_n), l(_l), max_depth(_max_depth), idx(_idx), partition_color(_partition_color), actual_max_depth(_actual_max_depth), tile_height(_tile_height)
    {
        if (_actual_max_depth == 0) {
            actual_max_depth = _max_depth;
        }
    }

};


struct TreeArgs{
    int value;
    bool is_leaf;
    TreeArgs( int _value, bool _is_leaf ) : value(_value), is_leaf(_is_leaf) {}
};

struct RefineHelperArgs
{
    int level;
    coord_t idx;
    RefineHelperArgs( int _level, coord_t _idx ) : level(_level), idx(_idx) {}
};

struct CompressHelperArgs{
    int level;
    coord_t idx;
    bool launch;
    int n;
    bool is_valid_entry;
    CompressHelperArgs( int _level, coord_t _idx, bool _launch, int _n , bool _is_valid_entry ) : level(_level), idx(_idx), launch(_launch), n(_n), is_valid_entry( _is_valid_entry ) {}
};



void print_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctxt, HighLevelRuntime *runtime) {
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    const FieldAccessor<READ_ONLY,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > read_acc(regions[0], FID_X);
    int node_counter=0;
    int max_depth = args.max_depth;
    queue<Arguments>tree;
    tree.push(args);
    while( !tree.empty() ){
        Arguments temp = tree.front();
        tree.pop();
        int n = temp.n;
        int l = temp.l;
        coord_t idx = temp.idx;
        coord_t idx_left_sub_tree = idx+1;
        coord_t idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));
        node_counter++;
        cout<<node_counter<<": "<<n<<"~"<<l<<"~"<<idx<<"~"<<read_acc[idx].value<<endl;
        if(!read_acc[idx].is_leaf){
            Arguments for_left_sub_tree (n + 1, l * 2    , max_depth, idx_left_sub_tree, temp.partition_color, temp.actual_max_depth, temp.tile_height);
            Arguments for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree, temp.partition_color, temp.actual_max_depth, temp.tile_height);
            tree.push( for_left_sub_tree );
            tree.push( for_right_sub_tree );
        }
    }
}



void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {

    int overall_max_depth = 8;
    int actual_left_depth = 3;
    int tile_height = 3;

    long int seed = 12345;
    {
        const InputArgs &command_args = HighLevelRuntime::get_input_args();
        for (int idx = 1; idx < command_args.argc; ++idx)
        {
            if (strcmp(command_args.argv[idx], "-max_depth") == 0)
                overall_max_depth = atoi(command_args.argv[++idx]);
            else if (strcmp(command_args.argv[idx], "-seed") == 0)
                seed = atol(command_args.argv[++idx]);
            else if(strcmp(command_args.argv[idx],"--tile") == 0)
                tile_height = atoi( command_args.argv[++idx]);
        }
    }
    srand(time(NULL));
    Rect<1> tree_rect(0LL, static_cast<coord_t>(pow(2, overall_max_depth + 1)));
    IndexSpace is = runtime->create_index_space(ctx, tree_rect);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(TreeArgs), FID_X);
    }

    LogicalRegion lr1 = runtime->create_logical_region(ctx, is, fs);
    Color partition_color1 = 10;

    Arguments args1(0, 0, overall_max_depth, 0, partition_color1, actual_left_depth, tile_height);
    args1.gen = rand();
    cout<<"Launching Refine Task"<<endl;
    TaskLauncher refine_launcher(REFINE_INTER_TASK_ID, TaskArgument(&args1, sizeof(Arguments)));
    refine_launcher.add_region_requirement(RegionRequirement(lr1, WRITE_DISCARD, EXCLUSIVE, lr1));
    refine_launcher.add_field(0, FID_X);
    runtime->execute_task(ctx, refine_launcher);

    cout<<"Launching Print Task After Refine"<<endl;
    TaskLauncher print_launcher(PRINT_TASK_ID, TaskArgument(&args1, sizeof(Arguments)));
    RegionRequirement req3( lr1 , READ_ONLY, EXCLUSIVE, lr1 );
    req3.add_field(FID_X);
    print_launcher.add_region_requirement( req3 );
    runtime->execute_task(ctx, print_launcher);

    cout<<"Launching Compress Task"<<endl;
    TaskLauncher compress_launcher(COMPRESS_INTER_TASK_ID, TaskArgument(&args1, sizeof(Arguments)));
    compress_launcher.add_region_requirement(RegionRequirement(lr1, WRITE_DISCARD, EXCLUSIVE, lr1));
    compress_launcher.add_field(0, FID_X);
    runtime->execute_task(ctx, compress_launcher);

    cout<<"Launching Print Task After Compress"<<endl;
    runtime->execute_task(ctx, print_launcher);
}

void refine_intra_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){

    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    queue<Arguments>tree;
    tree.push(args);
    LogicalRegion lr = regions[0].get_logical_region();
    LogicalPartition lp = LogicalPartition::NO_PART;
    LogicalRegion my_sub_tree_lr = lr;
    coord_t idx_left_sub_tree = 0LL;
    coord_t idx_right_sub_tree = 0LL;
    int max_depth = args.max_depth;
    int tile_height = args.tile_height;
    int helper_counter=0;
    while(!tree.empty()){
        Arguments temp = tree.front();
        tree.pop();
        int n = temp.n;
        int l = temp.l;
        coord_t idx = temp.idx;
        idx_left_sub_tree = idx+1;
        idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));
        long int node_value=rand();
        node_value = node_value % 10 + 1;
        const FieldAccessor<WRITE_DISCARD,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > write_acc(regions[0], FID_X);
        if (node_value <= 3 || n == max_depth - 1) {
            write_acc[idx].value = node_value % 3 + 1;
            write_acc[idx].is_leaf =true;
        }
        else {
            write_acc[idx].value = 0;
        }
        if( (node_value > 3 )&&( n < max_depth ) ){
            if( (n % tile_height )==( tile_height-1 ) ){
                const FieldAccessor<WRITE_DISCARD,RefineHelperArgs,1,coord_t,Realm::AffineAccessor<RefineHelperArgs,1,coord_t> > write_acc(regions[1], FID_X);
                write_acc[helper_counter].level = l;
                write_acc[helper_counter].idx = idx;
                helper_counter++;
            }
            else{
                Arguments for_left_sub_tree (n + 1, l * 2    , max_depth, idx_left_sub_tree, temp.partition_color, temp.actual_max_depth, tile_height);
                Arguments for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree, temp.partition_color, temp.actual_max_depth, tile_height);
                tree.push( for_left_sub_tree );
                tree.push( for_right_sub_tree );
            }
        }
        
    }

}

void refine_inter_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {

    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    int tile_height = args.tile_height;
    LogicalRegion lr = regions[0].get_logical_region();
    int n = args.n;
    Rect<1> helper_Array(0LL, static_cast<coord_t>(pow(2, tile_height-1)));
    IndexSpace is = runtime->create_index_space(ctx, helper_Array);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(RefineHelperArgs), FID_X);
    }
    LogicalRegion new_helper_Region = runtime->create_logical_region(ctx, is, fs);
    TaskLauncher refine_intra_launcher(REFINE_INTRA_TASK_ID, TaskArgument(&args, sizeof(Arguments) ) );
    RegionRequirement req1(lr, WRITE_DISCARD, EXCLUSIVE, lr);
    RegionRequirement req2(new_helper_Region, WRITE_DISCARD, EXCLUSIVE, new_helper_Region);
    req1.add_field(FID_X);
    req2.add_field(FID_X);
    refine_intra_launcher.add_region_requirement(req1);
    refine_intra_launcher.add_region_requirement(req2);
    runtime->execute_task(ctx,refine_intra_launcher);
    ArgumentMap arg_map;
    PhysicalRegion physicalRegion = runtime->map_region( ctx, req2 );
    const FieldAccessor<READ_ONLY,RefineHelperArgs,1,coord_t,Realm::AffineAccessor<RefineHelperArgs,1,coord_t> > read_acc(physicalRegion, FID_X);
    int task_counter=0;
    vector<pair<coord_t,coord_t> >color_index;
    coord_t right_idx_add = pow(2, args.max_depth - (n + tile_height -1 ) );
    for( int i = 0 ; i < (1<<(tile_height-1)); i++ ){
        coord_t idx = read_acc[i].idx;
        int level = read_acc[i].level;
        if( idx == 0 )
            break;
        coord_t idx_left_sub_tree = idx+1;
        coord_t idx_right_sub_tree = idx+right_idx_add;
        Arguments left_args( n + tile_height , 2*level , args.max_depth, idx_left_sub_tree , args.partition_color , args.actual_max_depth , args.tile_height);
        Arguments right_args( n + tile_height , 2*level+1 , args.max_depth, idx_right_sub_tree , args.partition_color, args.actual_max_depth, args.tile_height);
        arg_map.set_point( task_counter , TaskArgument(&left_args,sizeof(Arguments)));
        task_counter++;
        arg_map.set_point( task_counter, TaskArgument(&right_args, sizeof(Arguments)));
        task_counter++;
        color_index.push_back(make_pair(idx_left_sub_tree,idx_right_sub_tree-1));
        color_index.push_back(make_pair(idx_right_sub_tree, idx_right_sub_tree + right_idx_add - 2));
    }
    if( task_counter > 0 ){
        IndexSpace is = lr.get_index_space();
        DomainPointColoring coloring;
        for( int i = 0 ; i < task_counter ; i++ ){
            coloring[i]= Rect<1>(color_index[i].first,color_index[i].second);
        }
        Rect<1>color_space = Rect<1>(0,task_counter-1);
        IndexPartition ip = runtime->create_index_partition(ctx, is, color_space, coloring, DISJOINT_KIND, args.partition_color);
        LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);
        Rect<1> launch_domain(0,task_counter-1);
        IndexTaskLauncher refine_launcher(REFINE_INTER_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
        refine_launcher.add_region_requirement(RegionRequirement(lp,0,WRITE_DISCARD, EXCLUSIVE, lr));
        refine_launcher.add_field(0, FID_X);
        runtime->execute_index_space(ctx, refine_launcher);
    }
}


void compress_intra_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    queue<Arguments>tree;
    tree.push(args);
    coord_t idx_left_sub_tree = 0LL;
    coord_t idx_right_sub_tree = 0LL;
    int max_depth = args.max_depth;
    int tile_height = args.tile_height;
    int helper_counter=0;
    const FieldAccessor<READ_ONLY,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > read_acc(regions[0], FID_X);
    while(!tree.empty()){
        Arguments temp = tree.front();
        tree.pop();
        int n = temp.n;
        int l = temp.l;
        coord_t idx = temp.idx;
        idx_left_sub_tree = idx+1;
        idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));
        const FieldAccessor<WRITE_DISCARD,CompressHelperArgs,1,coord_t,Realm::AffineAccessor<CompressHelperArgs,1,coord_t> > write_acc(regions[1], FID_X);
        write_acc[helper_counter].level = l;
        write_acc[helper_counter].idx = idx;
        write_acc[helper_counter].n = n;
        write_acc[helper_counter].is_valid_entry = true;
        if( ((n % tile_height ) ==( tile_height-1 ) && ( !read_acc[idx].is_leaf ) ) )
            write_acc[helper_counter].launch = true;
        else if( !read_acc[idx].is_leaf ){
                Arguments for_left_sub_tree (n + 1, l * 2    , max_depth, idx_left_sub_tree, temp.partition_color, temp.actual_max_depth, tile_height);
                Arguments for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree, temp.partition_color, temp.actual_max_depth, tile_height);
                tree.push( for_left_sub_tree );
                tree.push( for_right_sub_tree );
        }
        helper_counter++;
    }
}


void compress_inter_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    int tile_height = args.tile_height;
    LogicalRegion lr = regions[0].get_logical_region();
    Rect<1> helper_Array(0LL, static_cast<coord_t>(pow(2, tile_height)-1));
    IndexSpace is = runtime->create_index_space(ctx, helper_Array);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(CompressHelperArgs), FID_X);
    }
    LogicalRegion new_helper_Region = runtime->create_logical_region(ctx, is, fs);
    TaskLauncher compress_intra_launcher(COMPRESS_INTRA_TASK_ID, TaskArgument(&args, sizeof(Arguments)));
    RegionRequirement req1(lr, READ_ONLY, EXCLUSIVE, lr);
    RegionRequirement req2(new_helper_Region, WRITE_DISCARD, EXCLUSIVE, new_helper_Region);
    req1.add_field(FID_X);
    req2.add_field(FID_X);
    compress_intra_launcher.add_region_requirement( req1 );
    compress_intra_launcher.add_region_requirement( req2 );
    runtime->execute_task(ctx,compress_intra_launcher);
    PhysicalRegion physicalRegion = runtime->map_region( ctx, req2 );
    const FieldAccessor<READ_ONLY,CompressHelperArgs,1,coord_t,Realm::AffineAccessor<CompressHelperArgs,1,coord_t> > read_acc(physicalRegion, FID_X);
    ArgumentMap arg_map;
    int task_counter=0;
    for( int  i = 0 ; i < (1<<tile_height) ; i++){
        bool launch = read_acc[i].launch;
        coord_t idx = read_acc[i].idx;
        int level = read_acc[i].level;
        int nx = read_acc[i].n;
        if( launch ){
            coord_t idx_left_sub_tree = idx+1;
            coord_t idx_right_sub_tree = idx+ static_cast<coord_t>(pow(2, args.max_depth - nx ));
            Arguments left_args( nx + 1 , 2*level , args.max_depth, idx_left_sub_tree , args.partition_color , args.actual_max_depth , args.tile_height);
            Arguments right_args( nx + 1 , 2*level+1 , args.max_depth, idx_right_sub_tree , args.partition_color, args.actual_max_depth, args.tile_height);
            arg_map.set_point( task_counter , TaskArgument(&left_args,sizeof(Arguments)));
            task_counter++;
            arg_map.set_point( task_counter, TaskArgument(&right_args, sizeof(Arguments)));
            task_counter++;
        }
    }
    if( task_counter > 0 ){
        LogicalPartition lp = runtime->get_logical_partition_by_color(ctx, lr, args.partition_color);
        Rect<1> launch_domain(0,task_counter-1);
        IndexTaskLauncher compress_launcher(COMPRESS_INTER_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
        compress_launcher.add_region_requirement(RegionRequirement(lp,0,WRITE_DISCARD, EXCLUSIVE, lr));
        compress_launcher.add_field(0, FID_X);
        runtime->execute_index_space(ctx, compress_launcher);
    }
    const FieldAccessor<WRITE_DISCARD,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > write_acc(regions[0], FID_X);
    for( int i = (1<<tile_height)-1; i>=0 ; i-- ){
        if( !read_acc[i].is_valid_entry )
            continue;
        coord_t idx = read_acc[i].idx;
        if( write_acc[idx].is_leaf )
            continue;
        int nx = read_acc[i].n;
        coord_t idx_left_sub_tree = idx+1;
        coord_t idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, args.max_depth - nx));
        write_acc[idx].value = write_acc[idx_left_sub_tree].value + write_acc[idx_right_sub_tree].value;
    }
}



int main(int argc, char** argv){

    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }

    {
        TaskVariantRegistrar registrar(REFINE_INTER_TASK_ID, "refine_inter");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<refine_inter_task>(registrar, "refine_inter");
    }

    {
        TaskVariantRegistrar registrar(REFINE_INTRA_TASK_ID, "refine_intra");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<refine_intra_task>(registrar, "refine_intra");
    }

    {
        TaskVariantRegistrar registrar(PRINT_TASK_ID, "print");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<print_task>(registrar, "print");
    }

    {
        TaskVariantRegistrar registrar(COMPRESS_INTER_TASK_ID, "compress_inter");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<compress_inter_task>(registrar, "compress_inter");
    }

    {
        TaskVariantRegistrar registrar(COMPRESS_INTRA_TASK_ID, "compress_intra");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<compress_intra_task>(registrar, "compress_intra");
    }

   
    return Runtime::start(argc,argv);
}
