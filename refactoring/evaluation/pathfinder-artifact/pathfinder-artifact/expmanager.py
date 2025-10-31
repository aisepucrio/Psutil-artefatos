from codecarbon import EmissionsTracker
import psutil
import csv
import os
import pathlib
import shutil
import time
import subprocess
from pathlib import Path
from functools import reduce
from utils import *
from dll_info import *


class FuzzerJob:
    def __init__(self,
                 image_name,
                 container_name,
                 container_working_dir,
                 container_env_vars,
                 host_fuzz_result_path,
                 container_fuzz_result_dir,
                 host_gcov_result_path,
                 container_cov_result_dir,
                 mem, use_conda):
        self.image_name = image_name
        self.container_name = container_name
        self.container_working_dir = container_working_dir
        self.container_env_vars = container_env_vars
        self.host_fuzz_result_path = host_fuzz_result_path
        self.container_fuzz_result_dir = container_fuzz_result_dir
        self.host_gcov_result_path = host_gcov_result_path
        self.container_cov_result_dir = container_cov_result_dir
        self.mem = mem
        self.use_conda = use_conda
        self.proc = None
        self.skip = False

    def check_dirs(self):
        pass

    def make_dirs(self):
        pass

    def _build_env_var_flag(self):
        return reduce(lambda acc, env_var: f"{acc} --env {env_var}", self.container_env_vars, "")

    def _build_volume_flag(self):
        volumes = []
        if self.host_fuzz_result_path and self.container_fuzz_result_dir:
            volumes.append(f"{self.host_fuzz_result_path}:{self.container_fuzz_result_dir}")
        if self.host_gcov_result_path and self.container_cov_result_dir:
            volumes.append(f"{self.host_gcov_result_path}:{self.container_cov_result_dir}")
        assert (len(volumes) > 0)
        return reduce(lambda acc, volume: f"{acc} -v {volume}", volumes, "")

    def command(self):
        pass

    def _build_docker_run_command(self):
        cmd = f"docker run -itd --cpus 1 -m {self.mem}g "
        cmd = f"{cmd} -w {self.container_working_dir}"
        cmd = f"{cmd} {self._build_env_var_flag()}"
        cmd = f"{cmd} --name {self.container_name}"
        cmd = f"{cmd} {self._build_volume_flag()}"
        cmd = f"{cmd} {self.image_name}"
        cmd = f"{cmd} sh -c \"{self.command()}\""
        return cmd

    def run(self):
        cmd = self._build_docker_run_command()
        print(cmd)
        self.proc = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)

    def wait(self):
        assert (self.proc is not None)  # should be called from started process
        return self.proc.wait()

    def container_exists(self):
        cmd = f"docker inspect {self.container_name}"
        self.proc = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        _ = self.proc.communicate()
        return self.proc.returncode == 0

    def is_running(self):
        cmd = f"docker inspect {self.container_name} --format='{{.State.Running}}'"
        self.proc = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, _ = self.proc.communicate()
        return out.decode().strip() == "true"

    def exitcode(self):
        cmd = f"docker inspect {self.container_name} --format='{{.State.ExitCode}}'"
        self.proc = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, _ = self.proc.communicate()
        out = out.decode().strip()
        if out.isdecimal():
            return int(out)
        else:
            return 1

    def stop(self):
        cmd = f"docker stop {self.container_name}"
        self.proc = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.proc.wait()

    def rm(self):
        cmd = f"docker rm {self.container_name}"
        self.proc = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.proc.wait()


class FuzzRunJob(FuzzerJob):
    def __init__(self,
                 image_name,
                 container_name,
                 container_working_dir,
                 container_env_vars,
                 host_fuzz_result_path,
                 container_fuzz_result_dir,
                 fuzzer_cmd,
                 time_budget,
                 mem, use_conda):
        super().__init__(image_name,
                         container_name,
                         container_working_dir,
                         container_env_vars,
                         host_fuzz_result_path,
                         container_fuzz_result_dir,
                         None,
                         None,
                         mem, use_conda)
        self.fuzzer_cmd = fuzzer_cmd
        self.time_budget = time_budget

    def check_dirs(self):
        if os.path.isdir(self.host_fuzz_result_path):
            print(f"Note: Skip `{self.container_name}`, as fuzz result exists in `{self.host_fuzz_result_path}`.")
            self.skip = True

    def make_dirs(self):
        os.makedirs(self.host_fuzz_result_path, exist_ok=True)
        host_corpus_dir = os.path.join(self.host_fuzz_result_path, "corpus")
        os.makedirs(host_corpus_dir, exist_ok=True)

    def loose_timeout(self):
        timeout_margin = 180
        return self.time_budget + timeout_margin

    def timeout_cmd(self):
        return f"timeout {self.loose_timeout()}"

    def command(self):
        cmd = f"{self.timeout_cmd()} {self.fuzzer_cmd}"
        if self.use_conda:
            cmd = f"conda run -n base {cmd}"
        return cmd

    def record_exitcode(self, exitcode):
        assert (os.path.isdir(self.host_fuzz_result_path))
        with open(os.path.join(self.host_fuzz_result_path, "exitcode.txt"), "w") as f:
            f.write(str(exitcode))


class GcovRunJob(FuzzerJob):
    def __init__(self,
                 image_name,
                 container_name,
                 container_working_dir,
                 container_env_vars,
                 host_fuzz_result_path,
                 container_fuzz_result_dir,
                 host_gcov_result_path,
                 container_cov_result_dir,
                 covrun_cmd_prefix,
                 itv_mode, itv_start, itv_end, job_id, apis,
                 mem, use_conda):
        super().__init__(image_name,
                         container_name,
                         container_working_dir,
                         container_env_vars,
                         host_fuzz_result_path,
                         container_fuzz_result_dir,
                         host_gcov_result_path,
                         container_cov_result_dir,
                         mem, use_conda)
        self.covrun_cmd_prefix = covrun_cmd_prefix
        self.itv_mode = itv_mode
        self.itv_start = itv_start
        self.itv_end = itv_end
        self.job_id = job_id
        self.apis = apis

    def check_dirs(self):
        if not os.path.isdir(self.host_fuzz_result_path):
            print(emphasize(f"Error: Result dir `{self.host_fuzz_result_path}` does not exists."))
            exit(0)
        if os.path.isdir(self.host_gcov_result_path):
            print(emphasize(f"Error: Result dir `{self.host_gcov_result_path}` exists."))
            exit(0)

    def make_dirs(self):
        os.makedirs(self.host_gcov_result_path, exist_ok=True)

    def job_id_str(self):
        return ("0" + str(self.job_id) if self.job_id < 10 else str(self.job_id))

    def out_file_name(self):
        return f"coverage{self.job_id_str()}.info"

    def out_file_rel_path(self):
        subdir = f"{self.itv_mode}{self.itv_start}_{self.itv_end}"
        return os.path.join(subdir, self.out_file_name())

    def command(self):
        apis_str = reduce(lambda a, b: f"{a} {b}", self.apis)

        cmd = self.covrun_cmd_prefix
        cmd = f"{cmd} --itv_mode {self.itv_mode}"
        cmd = f"{cmd} --itv_start {self.itv_start}"
        cmd = f"{cmd} --itv_end {self.itv_end}"
        cmd = f"{cmd} --apis {apis_str}"
        cmd = f"{cmd} --out {os.path.join(self.container_cov_result_dir, self.out_file_name())}"
        cmd = f"{cmd} > {self.container_cov_result_dir}/covrun_{self.job_id_str()}.log 2>&1"

        if self.use_conda:
            cmd = f"conda run -n base {cmd}"

        return cmd


class GcovMergeIntraJob(FuzzerJob):
    def __init__(self,
                 image_name,
                 container_name,
                 container_working_dir,
                 container_env_vars,
                 host_gcov_result_path,
                 container_cov_result_dir,
                 itv_subdir, job_id, intra_cov_jobs,
                 mem):
        super().__init__(image_name,
                         container_name,
                         container_working_dir,
                         container_env_vars,
                         None,
                         None,
                         host_gcov_result_path,
                         container_cov_result_dir,
                         mem, False)
        assert (len(intra_cov_jobs) >= 2)
        self.itv_subdir = itv_subdir
        self.job_id = job_id
        self.intra_cov_jobs = intra_cov_jobs

    def info_file_paths(self):
        paths = list(map(lambda intra_cov_job: intra_cov_job.out_file_name(), self.intra_cov_jobs))
        paths = list(map(lambda path: os.path.join(self.container_cov_result_dir, path), paths))
        return paths

    def job_id_str(self):
        return ("0" + str(self.job_id) if self.job_id < 10 else str(self.job_id))

    def out_file_name(self):
        return f"merged{self.job_id_str()}.info"

    def out_file_rel_path(self):
        return os.path.join(self.itv_subdir, self.out_file_name())

    def command(self):
        merge_flag = "--merge " + reduce(lambda a, b: f"{a} {b}", self.info_file_paths())

        cmd = f"{container_cov_tool_cmd()} --rm_input {merge_flag}"
        cmd = f"{cmd} --out {os.path.join(self.container_cov_result_dir, self.out_file_name())}"
        cmd = f"{cmd} > {self.container_cov_result_dir}/merge_intra_{self.job_id_str()}.log 2>&1"
        return cmd


class GcovMergeInterJob(FuzzerJob):
    def __init__(self,
                 image_name,
                 container_name,
                 container_working_dir,
                 container_env_vars,
                 host_gcov_result_path,
                 container_cov_result_dir,
                 intra_cov_jobs,
                 mem):
        super().__init__(image_name,
                         container_name,
                         container_working_dir,
                         container_env_vars,
                         None,
                         None,
                         host_gcov_result_path,
                         container_cov_result_dir,
                         mem, False)
        self.intra_cov_jobs = intra_cov_jobs

    def out_file_name(self):
        return f"merged.info"

    def info_file_paths(self):
        paths = list(map(lambda intra_cov_job: intra_cov_job.out_file_rel_path(), self.intra_cov_jobs))
        paths = list(map(lambda path: os.path.join(self.container_cov_result_dir, path), paths))
        return paths

    def command(self):
        merge_flag = "--merge " + reduce(lambda a, b: f"{a} {b}", self.info_file_paths())

        cmd = f"{container_cov_tool_cmd()} {merge_flag} --show_each"
        cmd = f"{cmd} --out {os.path.join(self.container_cov_result_dir, self.out_file_name())}"
        cmd = f"{cmd} > {self.container_cov_result_dir}/merge_inter.log 2>&1"
        return cmd


class GcovGenHtmlJob(FuzzerJob):
    def __init__(self,
                 image_name,
                 container_name,
                 container_working_dir,
                 container_env_vars,
                 host_gcov_result_path,
                 container_cov_result_dir,
                 inter_cov_job,
                 mem):
        super().__init__(image_name,
                         container_name,
                         container_working_dir,
                         container_env_vars,
                         None,
                         None,
                         host_gcov_result_path,
                         container_cov_result_dir,
                         mem, False)
        self.inter_cov_job = inter_cov_job

    def info_file_path(self):
        return os.path.join(self.container_cov_result_dir, self.inter_cov_job.out_file_name())

    def command(self):
        genhtml_flag = "--genhtml " + self.info_file_path()
        out_dir_path = os.path.join(self.container_cov_result_dir, f"html")

        cmd = f"{container_cov_tool_cmd()} {genhtml_flag}"
        cmd = f"{cmd} --out {out_dir_path}"
        cmd = f"{cmd} > {self.container_cov_result_dir}/genhtml.log 2>&1"
        return cmd


class JobScheduler:
    def __init__(self,
                 dll_info,
                 ablation,
                 time_budget,
                 repetition_indexes,
                 apis,
                 mem):
        self.dll_info = dll_info
        self.ablation = ablation
        self.time_budget = time_budget
        self.repetition_indexes = repetition_indexes
        self.apis = apis
        self.mem = mem

    def set_repeatition_indexes(self):
        repeat_max = 5

        if len(self.repetition_indexes) == 0:
            self.repetition_indexes = range(repeat_max)

    def schedule_worker(self):
        pass

    def schedule(self):
        self.set_repeatition_indexes()

        generations = self.schedule_worker()
        for generation in generations:
            for job in generation:
                job.check_dirs()
        return generations


class FuzzJobScheduler(JobScheduler):
    def __init__(self,
                 dll_info,
                 asan,
                 ablation,
                 time_budget,
                 repetition_indexes,
                 apis,
                 mem):
        super().__init__(dll_info,
                         ablation,
                         time_budget,
                         repetition_indexes,
                         apis,
                         mem)
        self.asan = asan
        if self.asan:
            self.dockerfile_name = self.dll_info.dockerfile_name("asan")
            self.image_name = self.dll_info.image_name("asan")
            self.host_result_base = os.path.join(root_asan_result_path(),
                                                 self.dll_info.host_asan_result_dir(self.ablation, self.time_budget))
            self.container_working_dir = self.dll_info.container_working_dir("asan")
            self.container_env_vars = self.dll_info.container_env_vars("asan")
        else:
            self.dockerfile_name = self.dll_info.dockerfile_name("fuzz")
            self.image_name = self.dll_info.image_name("fuzz")
            self.host_result_base = os.path.join(root_fuzz_result_path(),
                                                  self.dll_info.host_fuzz_result_dir(self.ablation, self.time_budget))
            self.container_working_dir = self.dll_info.container_working_dir("fuzz")
            self.container_env_vars = self.dll_info.container_env_vars("fuzz")
        self.use_conda = self.dll_info.use_conda

    def fuzzrun_job(self, repetition_index, api):
        ablation_str = f"-{self.ablation}" if self.ablation != "default" else ""
        container_name = f"pf-{self.dockerfile_name}{ablation_str}-{self.time_budget}sec-{repetition_index}-{api}"
        host_fuzz_result_path = os.path.join(self.host_result_base, f"{repetition_index}", api)

        fuzzer_cmd = f"{self.dll_info.fuzzer_cmd_prefix(self.ablation)}{api} {self.dll_info.fuzzer_flag(self.asan, self.ablation, self.time_budget)}"
        return FuzzRunJob(self.image_name,
                          container_name,
                          self.container_working_dir,
                          self.container_env_vars,
                          host_fuzz_result_path,
                          self.dll_info.container_fuzz_result_dir,
                          fuzzer_cmd,
                          self.time_budget,
                          self.mem, self.use_conda)

    def schedule_worker(self):
        jobs = []
        for repetition_index in self.repetition_indexes:
            for api in self.apis:
                jobs.append(self.fuzzrun_job(repetition_index, api))
        return [jobs]


class GcovJobScheduler(JobScheduler):
    def __init__(self,
                 dll_info,
                 ablation,
                 time_budget,
                 itv, itv_total, vs,
                 gen_html,
                 repetition_indexes,
                 apis,
                 cpu_capacity, mem):
        super().__init__(dll_info,
                         ablation,
                         time_budget,
                         repetition_indexes,
                         apis,
                         mem)
        assert (itv_total % itv == 0)
        assert (cpu_capacity is not None)

        self.itv_mode = "time"
        self.itv = itv
        self.itv_total = itv_total
        self.vs = vs
        self.gen_html = gen_html
        self.cpu_capacity = cpu_capacity

        self.dockerfile_name = self.dll_info.dockerfile_name("gcov")
        self.image_name = self.dll_info.image_name("gcov")
        self.host_fuzz_result_dir = self.dll_info.host_fuzz_result_dir(self.ablation, self.time_budget)
        self.host_gcov_result_dir = self.dll_info.host_gcov_result_dir(self.ablation, self.time_budget,
                                                                       self.itv, self.itv_total, self.vs)
        self.container_fuzz_result_dir = self.dll_info.container_fuzz_result_dir
        self.container_cov_result_dir = self.dll_info.container_cov_result_dir
        self.container_working_dir = self.dll_info.container_working_dir("gcov")
        self.container_env_vars = self.dll_info.container_env_vars("gcov")
        self.use_conda = self.dll_info.use_conda

    def itv_str(self, itv_start, itv_end):
        return f"{self.itv_mode}{itv_start}_{itv_end}"

    def covrun_job(self, repetition_index, job_id, apis,
                   itv_start, itv_end):
        ablation_str = f"-{self.ablation}" if self.ablation != "default" else ""
        vs_str = f"-vs_{self.vs}" if self.vs != None else ""
        container_name = f"pf-gcovrun-{self.dockerfile_name}{ablation_str}{vs_str}-{self.time_budget}sec-{repetition_index}-{self.itv_str(itv_start, itv_end)}-{job_id}"
        host_fuzz_result_path = os.path.join(root_fuzz_result_path(),
                                             self.host_fuzz_result_dir,
                                             f"{repetition_index}")
        host_gcov_result_path = os.path.join(root_gcov_result_path(),
                                            self.host_gcov_result_dir,
                                            f"{repetition_index}", self.itv_str(itv_start, itv_end))

        runner = os.path.join(container_home_path(), "pathfinder_coverage.py")
        covrun_cmd_prefix = f"python3 -u {runner}"
        covrun_cmd_prefix = f"{covrun_cmd_prefix} --cmd_prefix '{self.dll_info.fuzzer_cmd_prefix(self.ablation)}'"
        covrun_cmd_prefix = f"{covrun_cmd_prefix} --target_dir {self.dll_info.gcov_target_dir()}"
        covrun_cmd_prefix = f"{covrun_cmd_prefix} --third_party_dir {self.dll_info.third_party_dir()}"

        return GcovRunJob(self.image_name,
                          container_name,
                          self.container_working_dir,
                          self.container_env_vars,
                          host_fuzz_result_path,
                          self.container_fuzz_result_dir,
                          host_gcov_result_path,
                          self.container_cov_result_dir,
                          covrun_cmd_prefix,
                          self.itv_mode, itv_start, itv_end, job_id, apis,
                          self.mem, self.use_conda)

    def merge_intra_job(self, repetition_index, job_id,
                        itv_start, itv_end, intra_cov_jobs):
        ablation_str = f"-{self.ablation}" if self.ablation != "default" else ""
        vs_str = f"-vs_{self.vs}" if self.vs != None else ""
        container_name = f"pf-mergeintra-{self.dockerfile_name}{ablation_str}{vs_str}-{self.time_budget}sec-{repetition_index}-{self.itv_str(itv_start, itv_end)}-{job_id}"
        container_env_vars = []
        host_gcov_result_path = os.path.join(root_gcov_result_path(),
                                            self.host_gcov_result_dir,
                                            f"{repetition_index}", self.itv_str(itv_start, itv_end))
        itv_subdir = self.itv_str(itv_start, itv_end)
        return GcovMergeIntraJob(self.image_name,
                                 container_name,
                                 self.container_working_dir,
                                 container_env_vars,
                                 host_gcov_result_path,
                                 self.container_cov_result_dir,
                                 itv_subdir, job_id, intra_cov_jobs,
                                 self.mem)

    def merge_inter_job(self, repetition_index, intra_cov_jobs):
        ablation_str = f"-{self.ablation}" if self.ablation != "default" else ""
        vs_str = f"-vs_{self.vs}" if self.vs != None else ""
        container_name = f"pf-mergeinter-{self.dockerfile_name}{ablation_str}{vs_str}-{self.time_budget}sec-{repetition_index}"
        container_env_vars = []
        host_gcov_result_path = os.path.join(root_gcov_result_path(),
                                            self.host_gcov_result_dir,
                                            f"{repetition_index}")
        return GcovMergeInterJob(self.image_name,
                                 container_name,
                                 self.container_working_dir,
                                 container_env_vars,
                                 host_gcov_result_path,
                                 self.container_cov_result_dir,
                                 intra_cov_jobs,
                                 self.mem)

    def genhtml_job(self, repetition_index, inter_cov_job):
        ablation_str = f"-{self.ablation}" if self.ablation != "default" else ""
        vs_str = f"-vs_{self.vs}" if self.vs != None else ""
        container_name = f"pf-genhtml-{self.dockerfile_name}{ablation_str}{vs_str}-{self.time_budget}sec-{repetition_index}"
        container_env_vars = []
        host_gcov_result_path = os.path.join(root_gcov_result_path(),
                                            self.host_gcov_result_dir,
                                            f"{repetition_index}")
        return GcovGenHtmlJob(self.image_name,
                              container_name,
                              self.container_working_dir,
                              container_env_vars,
                              host_gcov_result_path,
                              self.container_cov_result_dir,
                              inter_cov_job,
                              self.mem)

    def num_interval(self):
        return self.itv_total // self.itv

    def jobs_per_itv_optimal(self):
        if (self.cpu_capacity < self.num_interval()):
            return

        n_apis = len(self.apis)
        n_jobs_per_itv = self.cpu_capacity // self.num_interval()
        max_apis_per_job = (n_apis // n_jobs_per_itv) + (0 if (n_apis % n_jobs_per_itv == 0) else 1)
        min_jobs_for_same_max_apis_per_job = (n_apis // max_apis_per_job) + (0 if (n_apis % max_apis_per_job == 0) else 1)
        return min_jobs_for_same_max_apis_per_job

    def __schedule_covrun_jobs_few_cores(self, repetition_index):
        n_core = self.cpu_capacity
        n_itv = self.num_interval()
        assert (n_core < n_itv)

        jobs_list = []

        for itv_idx in range(n_itv):
            itv_start = itv_idx * self.itv
            itv_end = itv_start + self.itv
            job_id = 0  # `job_id` is for distinguish jobs in the same interval.
            job = self.covrun_job(repetition_index, job_id, self.apis, itv_start, itv_end)
            jobs_list.append([job])

        return jobs_list

    def schedule_covrun_jobs(self, repetition_index):
        if self.cpu_capacity < self.num_interval():
            return self.__schedule_covrun_jobs_few_cores(repetition_index)

        jobs_list = []

        n_itv = self.itv_total // self.itv
        jobs_per_itv = self.jobs_per_itv_optimal()
        api_list = list(self.apis)

        for itv_idx in range(n_itv):
            api_idx_next = 0
            jobs = []
            for job_id in range(jobs_per_itv):
                n_apis = (len(api_list) // jobs_per_itv) + (1 if job_id <= (len(api_list) % jobs_per_itv - 1) else 0)
                apis = api_list[api_idx_next:api_idx_next + n_apis]
                api_idx_next += n_apis

                itv_start = itv_idx * self.itv
                itv_end = itv_start + self.itv
                job = self.covrun_job(repetition_index, job_id, apis, itv_start, itv_end)
                jobs.append(job)
                job_id += 1
            jobs_list.append(jobs)
        return jobs_list

    def schedule_merge_intra_generations(self, repetition_index, itv_idx, covrun_jobs):
        itv_start = itv_idx * self.itv
        itv_end = itv_start + self.itv

        generations = []

        generation_curr = covrun_jobs
        generations.append(generation_curr)

        job_id = 0
        unpaired = None
        while len(generation_curr) >= 2 or (len(generation_curr) == 1 and unpaired is not None):
            generation_next = []
            for i in range(0, len(generation_curr), 2):
                if i + 1 < len(generation_curr):
                    intra_cov_jobs = [generation_curr[i], generation_curr[i + 1]]
                    job = self.merge_intra_job(repetition_index, job_id, itv_start, itv_end, intra_cov_jobs)
                    job_id += 1
                    generation_next.append(job)
                else:
                    assert (i == len(generation_curr) - 1)

                    if unpaired == None:
                        unpaired = generation_curr[i]
                    else:
                        intra_cov_jobs = [generation_curr[i], unpaired]
                        unpaired = None
                        job = self.merge_intra_job(repetition_index, job_id, itv_start, itv_end, intra_cov_jobs)
                        job_id += 1
                        generation_next.append(job)
            generation_curr = generation_next
            generations.append(generation_curr)
        assert (len(generation_curr) == 1)
        return generations, generation_curr[0]

    def schedule_worker(self):
        generations_total = []
        for repetition_index in self.repetition_indexes:
            generations = []
            itv_intra_cov_jobs = []

            covrun_jobs_list = self.schedule_covrun_jobs(repetition_index)
            for itv_idx, covrun_jobs in enumerate(covrun_jobs_list):
                merge_intra_generations, itv_intra_cov_job = self.schedule_merge_intra_generations(repetition_index, itv_idx,
                                                                                                 covrun_jobs)
                itv_intra_cov_jobs.append(itv_intra_cov_job)

                for i, merge_intra_generation in enumerate(merge_intra_generations):
                    if len(generations) <= i:
                        generations.append([])
                    generations[i] += merge_intra_generation

            merge_inter_job = self.merge_inter_job(repetition_index, itv_intra_cov_jobs)
            generations.append([merge_inter_job])

            if self.gen_html:
                genhtml_job = self.genhtml_job(repetition_index, merge_inter_job)
                generations.append([genhtml_job])

            generations_total += generations

        return generations_total


class ExpManager:
    def __init__(self,
                 dll_info,
                 mode,
                 asan,
                 ablation,
                 vs,
                 apis,
                 time_budget,
                 rep,
                 itv,
                 itv_total,
                 gen_html,
                 cpu_capacity,
                 mem):
