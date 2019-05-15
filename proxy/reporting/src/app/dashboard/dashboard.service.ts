import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable()
export class DashboardService {
    constructor(private http: HttpClient) { }

    getGlobalStatus() {
        return this.http.get("/api?q=getStatus&context=global");
    }

    getBalance() {
        return this.http.get("/api?q=getBalance");
    }

    getGlobalHashrateHistory() {
        return this.http.get("/api?q=getGlobalHashrateHistory");
    }

    getWorkersList() {
        return this.http.get("/api?q=getWorkersList");
    }
}
