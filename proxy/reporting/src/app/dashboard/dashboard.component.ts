import { Component, OnInit } from '@angular/core';
import * as Chartist from 'chartist';
import {DashboardService} from "./dashboard.service";
import * as moment from "moment";
import { interval } from "rxjs/observable/interval";

declare var $:any;

class GlobalStatus {
    constructor() {
        this.cblocks_hashrate = 0;
        this.gblocks_hashrate = 0;
        this.uptime = 0;
        this.uptimeHours = "";
        this.cblocks_shares = 0;
        this.gblocks_shares = 0;
        this.cblocks_rejects = 0;
        this.gblocks_rejects = 0;
        this.workers_count = 0;
        this.current_block = 0;
        this.cblocks_dl = 0;
        this.gblocks_dl = 0;
        this.blocks = 0;
        this.best_dl = 0;
    }
    cblocks_hashrate : number;
    gblocks_hashrate : number;
    uptime : number;
    cblocks_shares : number;
    gblocks_shares : number;
    cblocks_rejects : number;
    gblocks_rejects : number;
    workers_count : number;
    current_block : number;
    cblocks_dl : number;
    gblocks_dl : number;
    blocks : number;
    best_dl : number;
    uptimeHours : string;
}

class Wallet {
    constructor() {
        this.balance = 0;
        this.last24 = 0;
    }

    balance : number;
    last24 : number;
}

class HashrateHistory {
    constructor() {
        this.cblocks_hashrate = 0;
        this.gblocks_hashrate = 0;
        this.timestamp = 0;
    }
    cblocks_hashrate : number;
    gblocks_hashrate : number;
    timestamp : number;
}

class WorkerItem {
    constructor() {
        this.worker_name = "";
        this.cblocks_hashrate = 0;
        this.gblocks_hashrate = 0;
        this.uptime = 0;
    }
    worker_name : string;
    cblocks_hashrate : number;
    gblocks_hashrate : number;
    uptime : number;
}

declare interface WorkersTable {
    headerRow: string[];
    dataRows: string[][];
}

@Component({
    selector: 'dashboard-cmp',
    moduleId: module.id,
    templateUrl: 'dashboard.component.html'
})

export class DashboardComponent implements OnInit{
    public globalStatus : GlobalStatus;
    public wallet : Wallet;
    public today : string;
    public workers: WorkersTable;

    private oneSecondUpdater : any;
    private oneMinuteUpdater : any;
    private tenMinutesUpdater : any;

    constructor(private dashboardService: DashboardService) {}

    ngOnInit(){
        this.globalStatus = new GlobalStatus();
        this.wallet = new Wallet();
        this.today = moment().format("ddd, D MMMM");
        this.workers = {
            headerRow: [ '#', 'Uptime', 'Name', '(C) Hashrate', '(G) Hashrate' ],
            dataRows: []
        };

        this.oneSecondUpdater = interval(1000).subscribe(i => {
            this.globalStatus.uptime++;
            this.globalStatus.uptimeHours = (new Date(this.globalStatus.uptime * 1000)).toISOString().substr(11, 8);
        });
        this.oneMinuteUpdater = interval(60000).subscribe(i => {
            this.updateDashboardData();
            this.updateWorkerList();
        });
        this.tenMinutesUpdater = interval(600000).subscribe(i => {
            this.updateWalletData();
        });

        this.updateDashboardData();
        this.updateWorkerList();
        this.updateWalletData();
    }

    public ngOnDestroy() {
        this.oneSecondUpdater.unsubscribe();
        this.oneMinuteUpdater.unsubscribe();
        this.tenMinutesUpdater.unsubscribe();
    }

    updateDashboardData() {
        this.dashboardService.getGlobalStatus().subscribe((data : GlobalStatus) => {
            this.globalStatus = data;
            this.globalStatus.uptimeHours = (new Date(data.uptime * 1000)).toISOString().substr(11, 8);
        });

        this.dashboardService.getGlobalHashrateHistory().subscribe((data : HashrateHistory[]) => {
            var series_cblocks = [];
            var series_gblocks = [];
            var labels = [];

            var date = new Date(); var timestamp = (date.getTime() / 1000) - 86400; //one day data

            var oldestDate = (data.length > 0) ? data[0].timestamp : (date.getTime() / 1000);

            for(var i = oldestDate - 600; i > timestamp; i-= 600) {
                series_cblocks.unshift({ x: new Date(i * 1000), y: 0 });
                series_gblocks.unshift({ x: new Date(i * 1000), y: 0 });
            }

            for(var i = 0; i<data.length;i++) {
                series_cblocks.push({ x: new Date(data[i].timestamp * 1000), y: data[i].cblocks_hashrate });
                series_gblocks.push({ x: new Date(data[i].timestamp * 1000), y: data[i].gblocks_hashrate });
            }

            var chart_data = {
                series: [
                    { name: "cblocks", data : series_cblocks },
                    { name: "gblocks", data: series_gblocks }
                ]
            };

            var options = {
                axisX: {
                    divisor: 12,
                    type: Chartist.FixedScaleAxis,
                    labelInterpolationFnc: function (value) {
                        return moment(value).format('HH:mm');
                    },
                    showGrid: true
                },
                height: "245px",
                showLine: true,
                showPoint: false
            };

            var responsiveOptions: any[] = [
                ['screen and (max-width: 640px)', {
                    axisX: {
                        labelInterpolationFnc: function (value) {
                            return null;
                        }
                    }
                }]
            ];

            new Chartist.Line('#chartActivity', chart_data, options, responsiveOptions);
            this.today = moment().format("ddd, D MMMM");
        });
    }

    updateWalletData() {
        this.dashboardService.getBalance().subscribe((data : Wallet) => {
            this.wallet = data;
        });
    }

    updateWorkerList() {
        this.dashboardService.getWorkersList().subscribe((data : WorkerItem[]) => {
            this.workers.dataRows = [];

            for(var i=0;i<data.length;i++) {
                this.workers.dataRows.push( [ (i+1).toString(),
                    (new Date(data[i].uptime * 1000)).toISOString().substr(11, 8),
                    data[i].worker_name,
                    data[i].cblocks_hashrate.toFixed(1),
                    data[i].gblocks_hashrate.toFixed()
                ] );
            }
        });
    }
}

