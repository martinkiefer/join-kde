import Utils
import copy

class JoinNode:
    def __init__(self, left, left_col, right, right_col, jid):
        self.left_col = left_col
        self.right_col = right_col
        self.left = left
        self.right = right
        self.jid = jid

    def generateTableEstimateKernel(self, f, query, estimator, stats):
        self.left.generateTableEstimateKernel(f, query, estimator, stats)
        self.right.generateTableEstimateKernel(f, query, estimator, stats)

    def generateJoinEstimateKernelBottomUp(self,f,query,estimator):
        self.left.generateJoinEstimateKernelBottomUp(f,query,estimator)
        self.right.generateJoinEstimateKernelBottomUp(f,query,estimator)

        icols = Utils.generateInvariantColumns(query)
        jcols = Utils.generateJoinColumns(query)

        t1, c1 = self.left_col
        t2, c2 = self.right_col
        #Are we the root join?
        if estimator.estimator == "AGPUJKDE" or estimator.estimator == "AGPUJKDE_COUNT":
            if isinstance(self.left, SampleScan):
                #Load all required values
                for c in jcols[t1]:
                    print >>f, "    unsigned int val_t%s_c%s = t%s_c%s[offset+get_global_id(0)];" % (t1,c,t1,c)
                if icols[t1]:
                    print >>f, "    double c_t%s = inv_t%s[offset+get_global_id(0)];" % (t1,t1)

            #And now, we need to perform the binary search.
            if estimator.join_kernel == "Cat":
                print >> f, "%sunsigned int pos_t%s = binarySearch(t%s_c%s, val_t%s_c%s - limit_t%s_c%s_t%s_c%s, n_t%s);" % ("    "*self.jid,t2,t2,c2,t1,c1,t1,c1,t2,c2,t2)
                print >> f, "%swhile(pos_t%s < n_t%s && t%s_c%s[pos_t%s] == (unsigned int) (val_t%s_c%s+limit_t%s_c%s_t%s_c%s)){" % ("    "*self.jid,t2,t2,t2,c2,t2,t1,c1,t1,c1,t2,c2)
            else:
                print >> f, "%sunsigned int pos_t%s = binarySearch(t%s_c%s, val_t%s_c%s - limit_t%s_c%s_t%s_c%s, n_t%s);" % (
                "    " * self.jid, t2, t2, c2, t1, c1, t1, c1, t2, c2, t2)
                print >> f, "%swhile(pos_t%s < n_t%s && t%s_c%s[pos_t%s] <= val_t%s_c%s+limit_t%s_c%s_t%s_c%s){" % (
                "    " * self.jid, t2, t2, t2, c2, t2, t1, c1, t1, c1, t2, c2)

            #Load contributions and variables

            for col in jcols[t2]:
                print >> f, "%sunsigned int val_t%s_c%s = t%s_c%s[pos_t%s];" % ("    "*(self.jid+1),t2,col,t2,col,t2)
            if icols[t2]:
                print >> f, "%sdouble c_t%s = inv_t%s[pos_t%s];" % ("    "*(self.jid+1),t2,t2,t2)
        elif estimator.estimator == "GPUSample" or estimator.estimator == "GPUCorrelatedSample":
            if isinstance(self.left, SampleScan):
                #Load all required values
                for c in jcols[t1]:
                    print >>f, "    unsigned int val_t%s_c%s = t%s_c%s[offset+get_global_id(0)];" % (t1,c,t1,c)

            #And now, we need to perform the binary search.
            print >> f, "%sunsigned int pos_t%s = binarySearch(t%s_c%s, val_t%s_c%s, n_t%s);" % ("    "*self.jid,t2,t2,c2,t1,c1,t2)
            print >> f, "%swhile(pos_t%s < n_t%s && t%s_c%s[pos_t%s] == val_t%s_c%s){" % ("    "*self.jid,t2,t2,t2,c2,t2,t1,c1)
            for col in jcols[t2]:
                print >> f, "%sunsigned int val_t%s_c%s = t%s_c%s[pos_t%s];" % ("    "*(self.jid+1),t2,col,t2,col,t2)
        else:
            raise Exception("This estimator is not handled by the join graph.")

    def getSortColumnDict(self,dict):
        if isinstance(self.left, SampleScan):
            dict[self.left_col[0]] = self.left_col[1]
        dict[self.right_col[0]] = self.right_col[1]
        self.left.getSortColumnDict(dict)
        self.right.getSortColumnDict(dict)

    def generateJoinEstimateKernelTopDown(self,f,query):
        t1, c1 = self.left_col
        t2, c2 = self.right_col
        print >> f, "%spos_t%s++;" % ("    "*(self.jid+1),t2)
        print >> f, "%s}" % ("    "*self.jid)
        self.left.generateJoinEstimateKernelTopDown(f,query)
        self.right.generateJoinEstimateKernelTopDown(f,query)

    def generateTableCode(self, f, query, estimator, limit=0, cu_factor = 2048):
        # Joins have no business in table code generation
        self.left.generateTableCode(f, query, estimator, limit, cu_factor)
        self.right.generateTableCode(f, query, estimator, limit, cu_factor)
        return self

    def collectTableIDs(self):
        tids = []
        tids += self.left.collectTableIDs()
        tids += self.right.collectTableIDs()
        return tids

    def collectJoinPairs(self):
        t1, c1 = self.left_col
        t2, c2 = self.right_col
        tids = []
        tids += self.left.collectJoinPairs()
        tids += self.right.collectJoinPairs()
        return tids + [(t1,c1,t2,c2)]

    def __str__(self):
        return "join(%s = %s, %s, %s)" % (self.left_col, self.right_col, self.left, self.right)


class SampleScan:
    def __init__(self, table):
        self.table = table

    def getSortColumnDict(self, dict):
        return dict

    def collectJoinPairs(self):
        return []

    def generateJoinEstimateKernelTopDown(self,f,query):
        pass

    def generateJoinEstimateKernelBottomUp(self,f,query,estimator):
        pass

    def generateTableEstimateKernel(self, f, query, estimator, stats):
        ts, dvals = stats
        cols = Utils.generateInvariantColumns(query)

        if estimator.estimator == "AGPUJKDE" or estimator.estimator == "AGPUJKDE_COUNT":
            kernels = [estimator.kernels[self.table][index] for index in cols[self.table]]
            if len(kernels) != 0:
                print >> f, "__kernel void invk_t%s(" % self.table
                for i, k in enumerate(kernels):
                    if k == "GaussRange":
                        print >> f, "    __global unsigned int* c%s, T h%s, unsigned int  u%s, unsigned int l%s," % (i, i, i, i)
                    elif k == "GaussPoint" or k == "CategoricalPoint":
                        print >> f, "    __global unsigned int* c%s, T h%s, unsigned int p%s," % (i, i, i)
                    else:
                        raise Exception("Unsupported kernel.")
                print >> f, "    __global T* o, unsigned int ss){"
                print >> f, "    for(unsigned int offset = 0; offset < ss; offset += get_global_size(0)){"
                print >> f, "        if(get_global_id(0)+offset < ss){"
                for i, k in enumerate(kernels):
                    if k == "GaussPoint":
                        print >> f, "            T ec%s = gaussPointEst(c%s[offset+get_global_id(0)], h%s, p%s);" % (i, i, i, i)
                    elif k == "GaussRange":
                        print >> f, "            T ec%s = gaussRangeEst(c%s[offset+get_global_id(0)], h%s, u%s, l%s);" % (
                        i, i, i, i, i)
                    elif k == "CategoricalPoint":
                        print >> f, "            T ec%s = catPointEst(c%s[offset+get_global_id(0)], h%s, p%s,%s);" % (
                        i, i, i, i, dvals[self.table][i])
                    else:
                        raise Exception("Unsupported kernel.")
                print >> f, "            o[offset+get_global_id(0)] = 1.0 ",
                for i, k in enumerate(kernels):
                    print >> f, "* ec%s" % i,
                print >> f, ";"
                print >> f, "       }"
                print >> f, "   }"
                print >> f, "}"
                print >> f

        elif estimator.estimator == "GPUSample" or estimator.estimator == "GPUCorrelatedSample":
            if len(cols[self.table]) != 0:
                print >> f, "__kernel void invk_t%s(" % self.table
                for i,j in enumerate(cols[self.table]):
                    if query.tables[self.table].columns[j].type == "range":
                        print >> f, "    __global unsigned int* c%s, unsigned int  u%s, unsigned int l%s," % (i, i, i)
                    elif query.tables[self.table].columns[j].type == "point":
                        print >> f, "    __global unsigned int* c%s, unsigned int p%s," % (i, i)
                    else:
                        raise Exception("Unsupported ctype.")
                print >> f, "    __global int* o, unsigned int ss){"
                print >> f, "    for(unsigned int offset = 0; offset < ss; offset += get_global_size(0)){"
                print >> f, "        if(get_global_id(0)+offset < ss) o[offset + get_global_id(0)+1] = 1 ",
                for i,j in enumerate(cols[self.table]):
                    if query.tables[self.table].columns[j].type == "range":
                        print >> f, " && c%s[offset + get_global_id(0)] != 0 && c%s[offset + get_global_id(0)] >= l%s && c%s[offset + get_global_id(0)] <= u%s " % (i,i, i, i,i),
                    elif query.tables[self.table].columns[j].type == "point":
                        print >> f, " && c%s[offset+get_global_id(0)] == p%s " % (i, i),
                    else:
                        raise Exception("Unsupported ctype.")
                print >> f, ";"
                print >> f, "    }"
                print >> f, "}"
                print >> f
        else:
            raise Exception("Join graph does not support this estimator.")

    def collectTableIDs(self):
        return [self.table]

    def generateTableCode(self, f, query, estimator, limit=0, cu_factor=2048):
        cols = Utils.generateInvariantColumns(query)
        jcols = Utils.generateJoinColumns(query)
        i = self.table
        indices = cols[i]
        jindices = jcols[i]

        if estimator.estimator == "AGPUJKDE" or estimator.estimator == "AGPUJKDE_COUNT":
            # Compute invariant contributions and return qualifying tuple ids.
            if len(indices) != 0:
                print >> f, "    size_t local%s = 64;" % i
                print >> f, "    size_t global%s = std::min((size_t) p->ctx.get_device().compute_units()*%s , ((p->ss%s-1)/local%s+1)*local%s);" % (i,cu_factor,i,i,i)
                print >> f, "    p->invk%s.set_args(" % i,
                for j in indices:
                    if estimator.kernels[i][j] == "GaussRange":
                        print >> f, "p->s_t%s_c%s, p->bw_t%s_c%s, u_t%s_c%s, l_t%s_c%s," % (i, j, i, j, i, j, i, j),
                    else:
                        print >> f, "p->s_t%s_c%s, p->bw_t%s_c%s, p_t%s_c%s," % (i, j, i, j, i, j),
                print >> f, " p->inv_t%s, (unsigned int) p->ss%s" % (i,i),
                print >> f, ");"
                print >> f, "    boost::compute::event ev%s = p->queue.enqueue_nd_range_kernel(p->invk%s,1,NULL,&(global%s), &(local%s));" % (i, i, i, i)
                # print >>f, "    ev%s.wait();" % i

                print >> f, "    boost::compute::transform(p->inv_t%s.begin(), p->inv_t%s.end(), p->count_t%s.begin()+1, boost::compute::lambda::_1 > %e, p->queue);" % (i, i, i, limit)
                print >> f, "    boost::compute::inclusive_scan(p->count_t%s.begin(), p->count_t%s.end(), p->map_t%s.begin(), p->queue);" % (i, i, i)
                print >> f, "    p->queue.finish();"
                print >> f, "    size_t rss_t%s = p->map_t%s[p->ss%s]+1;" % (i, i, i)
                print >> f, "    if(rss_t%s == 0) return 0.0;" % (i)
                for j in jindices:
                    print >> f, "    boost::compute::scatter_if(p->s_t%s_c%s.begin(), p->s_t%s_c%s.end(), p->map_t%s.begin()+1, p->count_t%s.begin()+1, p->sr_t%s_c%s.begin(), p->queue);" % (i,j,i,j,i,i,i,j)
                print >> f, "    boost::compute::scatter_if(p->inv_t%s.begin(), p->inv_t%s.end(),p->map_t%s.begin()+1, p->count_t%s.begin()+1, p->invr_t%s.begin(), p->queue);" % (i,i,i,i,i)
            else:
                # If there is no selection on this table, we don't have to do anything.
                print >> f, "    size_t rss_t%s = p->ss%s;" % (self.table,self.table)
            print >> f

        elif estimator.estimator == "GPUSample" or estimator.estimator == "GPUCorrelatedSample":
            # Compute invariant contributions and return qualifying tuple ids.
            if len(indices) != 0:
                print >> f, "    size_t local%s = 64;" % i
                print >> f, "    size_t global%s = std::min((size_t) p->ctx.get_device().compute_units()*%s , ((p->ss%s-1)/local%s+1)*local%s);" % (i,cu_factor,i,i,i)
                print >> f, "    p->invk%s.set_args(" % i,
                for j in indices:
                    if query.tables[i].columns[j].type == "range":
                        print >> f, "p->s_t%s_c%s, u_t%s_c%s, l_t%s_c%s," % (i, j, i, j, i, j),
                    elif query.tables[i].columns[j].type == "point":
                        print >> f, "p->s_t%s_c%s, p_t%s_c%s," % (i, j, i, j),
                    else:
                        raise Exception("Unsupported type.")
                print >> f, " p->count_t%s, (unsigned int) p->ss%s" % (i,i),
                print >> f, ");"
                print >> f, "    boost::compute::event ev%s = p->queue.enqueue_nd_range_kernel(p->invk%s,1,NULL,&global%s, &local%s);" % (i, i,i,i)
                # print >>f, "    ev%s.wait();" % i

                #print >> f, "    boost::compute::transform(p->inv_t%s.begin(), p->inv_t%s.end(), p->count_t%s.begin()+1, boost::compute::lambda::_1 > %e, p->queue);" % (i, i, i, limit)
                print >> f, "    boost::compute::inclusive_scan(p->count_t%s.begin(), p->count_t%s.end(), p->map_t%s.begin(), p->queue);" % (i, i, i)
                print >> f, "    p->queue.finish();"
                print >> f, "    size_t rss_t%s = p->map_t%s[p->ss%s]+1;" % (i, i, i)
                print >> f, "    if(rss_t%s == 0) return 0.0;" % (i)
                for j in jindices:
                    print >> f, "    boost::compute::scatter_if(p->s_t%s_c%s.begin(), p->s_t%s_c%s.end(), p->map_t%s.begin()+1, p->count_t%s.begin()+1, p->sr_t%s_c%s.begin(), p->queue);" % (i,j,i,j,i,i,i,j)
                print >> f, "    boost::compute::scatter_if(p->inv_t%s.begin(), p->inv_t%s.end(),p->map_t%s.begin()+1, p->count_t%s.begin()+1, p->invr_t%s.begin(), p->queue);" % (i,i,i,i,i)
            else:
                # If there is no selection on this table, we don't have to do anything.
                print >> f, "    size_t rss_t%s = p->ss%s;" % (self.table,self.table)
            print >> f
        else:
            raise Exception("The join graph does not handle this type of estimator.")


    def generateJoinCode(self, f, estimator, ts):
        # Sample scans have no business in join code generation
        return self

    def __str__(self):
        return "SS(%s)" % str(self.table)


def constructJoinGraph(query):
    tables = [set() for _ in query.joins]

    joins = copy.deepcopy(query.joins)

    for i, join in enumerate(query.joins):
        for j in join:
            tables[i].add(j[0])
    # Start off with the initial join
    jc = tuple(joins[0][0])
    tree = SampleScan(jc[0])

    jid = 1
    for j in joins[0][1:]:
        tree = JoinNode(tree, jc, SampleScan(j[0]), tuple(j), jid)
        jid += 1
    tree_tables = tables.pop(0)
    joins.pop(0)

    while joins:
        for i, (join, table) in enumerate(zip(joins, tables)):
            # Okay, this join works together with our previous join tree.
            if tree_tables & table:
                join = copy.deepcopy(join)
                col_left = None
                for k, tcol in enumerate(join):
                    x, y = tcol
                    # We found the column from the other join.
                    if x in tree_tables:
                        col_left = (x, y)
                        join.pop(k)
                        break
                if col_left is None:
                    raise Exception("Join tree construction failed. Does your query contain a cross?")

                jc = None
                for tcol in join:
                    x, y = tcol
                    tree = JoinNode(tree, col_left, SampleScan(x), (x, y), jid)
                    jid += 1

                tree_tables |= tables.pop(i)
                joins.pop(i)
                break
    return tree
