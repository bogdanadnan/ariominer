import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable()
export class WorkerService {
    constructor(private http: HttpClient) { }

    getWorkerStatus(workerId : string) {
        return this.http.get("/api?q=getStatus&context=" + workerId);
    }

    getWorkerHashrateHistory(workerId : string) {
        return this.http.get("/api?q=getWorkerHashrateHistory&workerId=" + workerId);
    }

    getWorkerDevices(workerId : string) {
        return this.http.get("/api?q=getWorkerDevices&workerId=" + workerId);
    }
}
