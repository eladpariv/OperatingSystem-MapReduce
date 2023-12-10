#include "MapReduceFramework.h"
#include "Barrier/Barrier.cpp"
#include <algorithm>
#include <atomic>
#include "Barrier/Barrier.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#define INITIALIZE_THREAD_ERROR "system error: thread initialize fail"
#define JOIN_TREAD_ERROR "system error: thread join fail"


typedef void* (*ThreadStartRoutine)(void*);
typedef struct{
    pthread_mutex_t mutex_of_reduce_phase;
    pthread_mutex_t emit2_mutex;
    pthread_mutex_t mutex_of_shuffle_phase;
    pthread_mutex_t reduceInitMutex;
    pthread_mutex_t forJobMutex;
    pthread_mutex_t emit3_mutex;
    pthread_mutex_t logPrintMutex;
    pthread_mutex_t phase_thread;
    pthread_mutex_t mutex_of_map_phase;

}mutex_struct;

typedef struct JobContext{
    std::vector<IntermediateVec> &queue;
    std::map<pthread_t,IntermediateVec*>& toVecMap; // maps thread ids to intermediate vectors
    std::map<K2*,IntermediateVec>& keyToVecMap; // maps k2 To vector of v2s (for shuffling)
    std::vector<IntermediateVec*>& memoryVec;
    OutputVec& outputVec;
    Barrier *barrier;
    pthread_t mainThread;
    const MapReduceClient &client;
    const InputVec& inputVec;
    mutex_struct mutexes;
    int threadsNum;
    bool waited;
    bool firstIter;
    pthread_t *threads;
    uint64_t curInterLen;
    std::atomic<uint64_t> percentageCounter;
    std::atomic<uint64_t> atomicCounter;
}JobContext;



const uint64_t UINT64_TWO = 2;
const uint64_t UINT64_SIXTY_TWO = 62;
const uint64_t UINT64_THIRTY_THREE = 33;
const uint64_t UINT64_ONE  =1;
const int NUM_MUTEXES = 9;
const uint64_t UINT64_THIRTY_ONE = 31;
const int64_t BITWISE_FIRST_31 = (uint64_t)1 << 31;
const char* lock_error_message[2] = {"system error: lock mutex failed","system error: unlock mutex failed"};

/**
 * locks and unlock a given mutex
 * @param mutex
 * @param flag 0 for lock and 1 for unlock
 */
void lock_handle(pthread_mutex_t* mutex,int flag){
    if(flag == 0){
        if (pthread_mutex_lock(mutex) != 0){
            std::cerr << lock_error_message[flag] << std::endl;
            exit(EXIT_FAILURE);
        }
    }else{
        if (pthread_mutex_unlock(mutex) != 0){
            std::cerr << lock_error_message[flag] << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}


/**
 * getter for the job state.
 * @param job the job to know
 * @param state - new state to get the data
 */

void getJobState(JobHandle job, JobState* state) {
    auto jc = reinterpret_cast<JobContext*>(job);
    unsigned long atomic_c = jc->percentageCounter.load();
    state->stage = static_cast<stage_t>((atomic_c >> UINT64_SIXTY_TWO) & 3);
    state->percentage = static_cast<float>(atomic_c % BITWISE_FIRST_31) / static_cast<float>((atomic_c << UINT64_TWO) >>
                                                                                                                      UINT64_THIRTY_THREE) * 100.0f;
}


/**
 * map phase algorithm
 * @param jc the job context to map
 */
void map_method(JobContext *jc) {
    uint64_t v= 0;
    int input_size = jc->inputVec.size();
    while (true){
        if((v & uint64_t(pow(2, 31) - 1)) >= input_size - 1){
            lock_handle(&jc->mutexes.mutex_of_map_phase,1);
            return;
        }
        lock_handle(&jc->mutexes.mutex_of_map_phase,0);
        v = jc->atomicCounter.load();
        if((v & uint64_t(pow(2, 31) - 1)) < input_size){
            jc->atomicCounter++;
            jc->percentageCounter += 1;
            InputPair p = jc->inputVec[(int) (v & uint64_t(pow(2, 31) - 1))];
            jc->client.map(p.first,p.second,jc);
            lock_handle(&jc->mutexes.mutex_of_map_phase,1);
        }
    }
}

/**
 * sorts the self thread vector after map stage
 * @param jc the job context
 *
 */
void sort_method(JobContext *jc) {
    IntermediateVec *curr_vec = jc->toVecMap[pthread_self()];
    std::sort(curr_vec->begin(),curr_vec->end());
}

/**
 * updating the percentage counter
 * @param jc the job context
 * @param job job id
 * @param currentStage the current stage of the job
 */
void increasePercentageCounter(JobContext* jc, const int job, const int currentStage){
    unsigned long long newPercent = (BITWISE_FIRST_31 * currentStage + job) << UINT64_THIRTY_ONE;
    jc->percentageCounter = newPercent;
}

/**
 * helper function of process_vec to create new vector with the given pair
 * @param jc job context
 * @param cur_pair the pair to push into the vector
 */
void CreateNewVector(JobContext *jc,IntermediatePair cur_pair){
    auto * new_vec = new IntermediateVec();
    jc->memoryVec.push_back(new_vec);
    new_vec->push_back(cur_pair);
    (jc->atomicCounter)++;  // count number of vectors in queue
    jc->keyToVecMap[cur_pair.first] = *new_vec;
}

/**
 * helper of shuffle method
 * @param jc  job context
 * @param cur_vec the vector to process
 */
void process_vec(JobContext *jc, IntermediateVec *cur_vec) {
    while(!cur_vec->empty()){
        IntermediatePair cur_pair = cur_vec->back();
        cur_vec->pop_back();
        bool founded = false;
        if(!jc->keyToVecMap.empty()){
            for(auto &elem: jc->keyToVecMap){
                if(!((*elem.first < *cur_pair.first)||(*cur_pair.first < *elem.first))){
                    jc->keyToVecMap[elem.first].push_back(cur_pair);
                    founded = true;
                }
            }
            if(!founded){

                CreateNewVector(jc,cur_pair);
            }
        }
        else{
            jc->keyToVecMap[cur_pair.first].push_back(cur_pair);
            (jc->atomicCounter)++;
            continue;
        }
    }
    jc->percentageCounter++;
}

/**
 * the shuffle method
 * @param jc job context
 */
void shuffle_method(JobContext *jc) {
    uint64_t input_size = jc->inputVec.size();

    if (pthread_self() != jc->mainThread) return;

    lock_handle(&jc->mutexes.mutex_of_shuffle_phase,0);
    increasePercentageCounter(jc,  jc->toVecMap.size(),2);
    jc->atomicCounter -= input_size;

    for(auto &tid: jc->toVecMap){
        IntermediateVec *cur_vec = tid.second;
        process_vec(jc, cur_vec);
    }

    increasePercentageCounter(jc,  jc->keyToVecMap.size(),3);
    lock_handle(&jc->mutexes.mutex_of_shuffle_phase,1);
}
/*
 * helper to update the jobContext in during reduce phase.
 */
void jobUpdateReduce(JobContext * jobContext,int size){
    IntermediateVec v = jobContext->queue[size-1];
    jobContext->curInterLen = v.size();
    jobContext->queue.pop_back();
    jobContext->percentageCounter ++;
    jobContext->atomicCounter += UINT64_ONE << UINT64_THIRTY_ONE;
    jobContext->client.reduce(&v,jobContext);
}
/**
 * the reduce method
 * @param jc job context
 */
void reduce_phase(JobContext *jobContext) {

    lock_handle(&jobContext->mutexes.mutex_of_reduce_phase,0);
    if (jobContext->firstIter){
        jobContext->firstIter = !jobContext->firstIter;
        uint64_t size = ((jobContext->atomicCounter << UINT64_TWO) >> UINT64_THIRTY_THREE) << UINT64_THIRTY_ONE;
        jobContext->atomicCounter -= size;
        for (auto &elem: jobContext->keyToVecMap){
            IntermediateVec v;
            for (auto &pair: elem.second){
                v.push_back(pair);
            }
            jobContext->queue.push_back(v);
        }
    }
    lock_handle(&jobContext->mutexes.mutex_of_reduce_phase,1);

    lock_handle(&jobContext->mutexes.mutex_of_reduce_phase,0);

    while((jobContext->atomicCounter << UINT64_TWO) >> UINT64_THIRTY_THREE <
    ((jobContext->atomicCounter)&uint64_t(pow(2, 31) - 1))){
        int q_size = jobContext->queue.size();
        if(q_size == 0){ break;}
        jobUpdateReduce(jobContext,q_size);
    }
    lock_handle(&jobContext->mutexes.mutex_of_reduce_phase,1);
}

/**
 * init struct of all mutexes
 * @param mutexes the mutexes
 * @return initialized mutex_struct
 */
mutex_struct mutex_struct_builder(pthread_mutex_t *mutexes){
    mutex_struct mutexStruct = {
            .mutex_of_reduce_phase = mutexes[0],
            .emit2_mutex = mutexes[1],
            .mutex_of_shuffle_phase = mutexes[2],
            .reduceInitMutex = mutexes[3],
            .forJobMutex = mutexes[4],
            .emit3_mutex = mutexes[5],
            .logPrintMutex = mutexes[6],
            .phase_thread = mutexes[7],
            .mutex_of_map_phase = mutexes[8]
    };
    return mutexStruct;
}

/**
 * create job
 */
JobContext * createJob(const MapReduceClient &client, const InputVec &inputVec, OutputVec &outputVec,
                       int multiThreadLevel,pthread_mutex_t *mutexes,pthread_t *threads){
    JobContext *job_c = new JobContext{
                    .queue = *new std::vector<IntermediateVec>,
                    .toVecMap = *new std::map<pthread_t, IntermediateVec *>,
                    .keyToVecMap = *new std::map<K2 *, IntermediateVec>,
                    .memoryVec = *new std::vector<IntermediateVec *>,
                    .outputVec = outputVec,
                    .barrier = new Barrier(multiThreadLevel),
                    .client = client,
                    .inputVec = inputVec,
                    .mutexes = mutex_struct_builder(mutexes),
                    .threadsNum = multiThreadLevel,
                    .waited = false,
                    .firstIter = true,
                    .threads = threads,
                    .percentageCounter{0},
                    .atomicCounter{0}
    };
    return job_c;
}
/**
 * create the full thread routine
 * @param jobContext
 * @return pointer to thread routine section
 */
ThreadStartRoutine createThreadStartRoutine(JobContext *jobContext) {
    return [](void *arg) -> void * {
        auto *jobContext = reinterpret_cast<JobContext *>(arg);
        lock_handle(&jobContext->mutexes.phase_thread, 0);
        auto *cur_vec = new std::vector<IntermediatePair>();
        if (jobContext->toVecMap.empty()) {
            jobContext->mainThread = pthread_self();
        }
        if (jobContext->percentageCounter >> 62 == 0) {
            jobContext->percentageCounter = (BITWISE_FIRST_31 + jobContext->inputVec.size()) << UINT64_THIRTY_ONE;
        }
        jobContext->toVecMap.insert({pthread_self(), cur_vec}); // adding zero thread
        lock_handle(&jobContext->mutexes.phase_thread, 1);
        map_method(jobContext);
        sort_method(jobContext);
        jobContext->barrier->barrier();
        shuffle_method(jobContext);
        jobContext->barrier->barrier();
        reduce_phase(jobContext);
        return nullptr;
    };
}

/**
 * run the all process of the job
 * @param client the client
 * @param inputVec
 * @param outputVec
 * @param multiThreadLevel
 * @return
 */
JobHandle startMapReduceJob(const MapReduceClient &client, const InputVec &inputVec, OutputVec &outputVec, int multiThreadLevel) {
    pthread_mutex_t mutexes[9];
    pthread_t *threads = reinterpret_cast<pthread_t *>(malloc(sizeof(pthread_t) * multiThreadLevel));
    for (int i = 0; i < NUM_MUTEXES; i++) {
        pthread_mutex_init(&mutexes[i], NULL);
    }
    JobContext *job_c = createJob(client, inputVec, outputVec, multiThreadLevel, mutexes, threads);
    pthread_mutex_t init_mutex = PTHREAD_MUTEX_INITIALIZER;
    lock_handle(&init_mutex, 0);
    ThreadStartRoutine threadStartRoutine = createThreadStartRoutine(job_c);
    for (int i = 0; i < multiThreadLevel; ++i) {
        if (pthread_create(&threads[i], nullptr, threadStartRoutine, job_c) != 0) {
            std::cerr << INITIALIZE_THREAD_ERROR << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    lock_handle(&init_mutex, 1);
    for (int i = 0; i < NUM_MUTEXES; i++) {
        pthread_mutex_destroy(&mutexes[i]);
    }
    auto jc = static_cast<JobHandle>(job_c);
    return jc;
}


/**
 * saves the intermediary elements and updates the number of the intermediary elements.
 * @param key
 * @param value
 * @param context
 */


void emit2 (K2* key, V2* value, void* context) {
    JobContext* jc = (JobContext*) context;
    pthread_t tid = pthread_self();

    lock_handle(&jc->mutexes.emit2_mutex, 0);

    IntermediateVec*& interVec = jc->toVecMap[tid];
    if (!interVec) {
        interVec = new IntermediateVec();
        jc->toVecMap[tid] = interVec;
    }

    interVec->emplace_back(IntermediatePair(key, value));

    lock_handle(&jc->mutexes.emit2_mutex, 1);

    jc->atomicCounter += uint64_t ((uint64_t)1 << 31);
}

/**
 * saves the output elements inside the outputVec.
 * @param key
 * @param value
 * @param context
 */
void emit3 (K3* key, V3* value, void* context){
    auto jc = (JobContext*) context;
    lock_handle(&jc->mutexes.emit3_mutex,0);
    jc->outputVec.push_back(OutputPair(key, value));
    lock_handle(&jc->mutexes.emit3_mutex,1);
}

/**
 * wait for jobs by using the pthread_join method.
 * @param job
 */
void waitForJob(JobHandle job) {
    auto jc = (JobContext *) job;
    lock_handle(&jc->mutexes.forJobMutex, 0);
    if(jc->waited){ return;}

    jc->waited = true;
    for (int i = 0; i < jc->threadsNum; ++i) {
        if (pthread_join(jc->threads[i], nullptr) !=0){
            std::cout << JOIN_TREAD_ERROR << " " << jc->threads[i] << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    lock_handle(&jc->mutexes.forJobMutex, 1);

}
/*
 * helper for closeJobHandel for  a cleaner code
 */
void deleteJobFields(JobContext * jobContext){
    delete &jobContext->memoryVec;
    delete &jobContext->toVecMap;
    delete &jobContext->keyToVecMap;
    delete &jobContext->queue;
    delete jobContext;
}

/**
 * destroy the mutexes and free the allocated memory
 * @param job
 */
void closeJobHandle(JobHandle job){
    waitForJob(job);
    auto jc = (JobContext*) job;
    delete jc->barrier;

    std::for_each(jc->memoryVec.begin(), jc->memoryVec.end(), [](IntermediateVec* elem) {
        delete elem;
    });
    for (auto &elem: jc->toVecMap){
        delete elem.second;
    }
    free(jc->threads); jc->threads = nullptr;
    deleteJobFields(jc);


}

