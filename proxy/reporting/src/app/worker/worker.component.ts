import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, Params } from '@angular/router';
import { WorkerService } from "./worker.service";
import {interval} from "rxjs/observable/interval";
import * as Chartist from 'chartist';
import * as moment from "moment";

class WorkerStatus {
    constructor() {
        this.cblocks_hashrate = 0;
        this.gblocks_hashrate = 0;
        this.uptime = 0;
        this.uptimeHours = "";
        this.cblocks_shares = 0;
        this.gblocks_shares = 0;
        this.cblocks_rejects = 0;
        this.gblocks_rejects = 0;
        this.devices_count = 0;
        this.blocks = 0;
    }
    cblocks_hashrate : number;
    gblocks_hashrate : number;
    uptime : number;
    cblocks_shares : number;
    gblocks_shares : number;
    cblocks_rejects : number;
    gblocks_rejects : number;
    devices_count : number;
    blocks : number;
    uptimeHours : string;
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

class DeviceItem {
    constructor() {
        this.hasher_name = "";
        this.device_name = "";
        this.cblocks_hashrate = 0;
        this.gblocks_hashrate = 0;
    }
    hasher_name : string;
    device_name : string;
    cblocks_hashrate : number;
    gblocks_hashrate : number;
}

declare interface DevicesTable {
    headerRow: string[];
    dataRows: string[][];
}

@Component({
    selector: 'table-cmp',
    moduleId: module.id,
    templateUrl: 'worker.component.html'
})

export class WorkerComponent implements OnInit{
    public workerId: string;
    public workerStatus: WorkerStatus;
    public today: string;
    public devices: DevicesTable;
    private oneSecondUpdater : any;
    private oneMinuteUpdater : any;


    constructor(private route: ActivatedRoute, private workerService: WorkerService) {}

    ngOnInit(){
        this.route.url.subscribe(params => {
            if(params.length == 2) {
                this.workerId = params[1].path;
                this.workerStatus = new WorkerStatus();
                this.today = moment().format("ddd, D MMMM");
                this.devices = {
                    headerRow: ['#', 'Hasher', 'Name', '(C) Hashrate', '(G) Hashrate'],
                    dataRows: []
                };

                this.updateWorkerStatus();
                this.updateWorkerDevices();
                this.updateWorkerHashrate();
            }
        });

        this.workerStatus = new WorkerStatus();
        this.today = moment().format("ddd, D MMMM");
        this.devices = {
            headerRow: [ '#', 'Hasher', 'Name', '(C) Hashrate', '(G) Hashrate' ],
            dataRows: []
        };

        this.route.params.forEach((urlParameters) => {
            this.workerId = urlParameters['id'];
        });

        this.oneSecondUpdater = interval(1000).subscribe(i => {
            this.workerStatus.uptime++;
            this.workerStatus.uptimeHours = (new Date(this.workerStatus.uptime * 1000)).toISOString().substr(11, 8);
        });
        this.oneMinuteUpdater = interval(60000).subscribe(i => {
            this.updateWorkerStatus();
            this.updateWorkerDevices();
            this.updateWorkerHashrate();
        });

        this.updateWorkerStatus();
        this.updateWorkerDevices();
        this.updateWorkerHashrate();
    }

    public ngOnDestroy() {
        this.oneSecondUpdater.unsubscribe();
        this.oneMinuteUpdater.unsubscribe();
    }

    updateWorkerStatus() {
        this.workerService.getWorkerStatus(this.workerId).subscribe((data : WorkerStatus) => {
            this.workerStatus = data;
            this.workerStatus.uptimeHours = (new Date(data.uptime * 1000)).toISOString().substr(11, 8);
        });
    }

    updateWorkerHashrate() {
        this.workerService.getWorkerHashrateHistory(this.workerId).subscribe((data : HashrateHistory[]) => {
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

    updateWorkerDevices() {
        this.workerService.getWorkerDevices(this.workerId).subscribe((data : DeviceItem[]) => {
            this.devices.dataRows = [];

            for(var i=0;i<data.length;i++) {
                this.devices.dataRows.push( [ (i+1).toString(),
                    data[i].hasher_name,
                    data[i].device_name,
                    data[i].cblocks_hashrate.toFixed(1),
                    data[i].gblocks_hashrate.toFixed()
                ] );
            }
        });
    }
}
