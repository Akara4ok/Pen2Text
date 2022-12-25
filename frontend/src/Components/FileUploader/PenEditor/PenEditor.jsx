import React, { createRef } from 'react';
import classes from './PenEditor.scss';
import Button from '@Components/Button/Button';
import { srcToFile } from '../../../utils/utils';
class PenEditor extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            size: 5,
            shouldDraw: false,
            theContext: undefined,
            drawnFiles: this.props.drawnFiles ?? 0,
        };
        this.canvasRef = createRef();
    }

    componentDidMount() {
        const canvasComponent = this.canvasRef.current;
        let dpr = window.devicePixelRatio || 1;
        let rect = canvasComponent.getBoundingClientRect();
        canvasComponent.width = rect.width * dpr;
        canvasComponent.height = rect.height * dpr;

        const theContext = canvasComponent.getContext('2d', {
            willReadFrequently: true,
        });
        theContext.scale(dpr, dpr);
        theContext.fillStyle = "white";
        theContext.fillRect(0, 0, canvasComponent.width, canvasComponent.height);

        this.setState({ theContext });
    }

    start = event => {
        const { theContext, size } = this.state;
        theContext.lineWidth = size;
        theContext.lineJoin = 'round';
        theContext.lineCap = 'round';
        theContext.beginPath();
        let elementRect = event.target.getBoundingClientRect();
        theContext.moveTo(
            event.clientX - elementRect.left,
            event.clientY - elementRect.top,
        );
        this.setState({ shouldDraw: true });
    };

    end = () => {
        this.setState({ shouldDraw: false });
    };

    move = event => {
        const { shouldDraw, theContext } = this.state;
        if (shouldDraw) {
            let elementRect = event.target.getBoundingClientRect();
            theContext.lineTo(
                event.clientX - elementRect.left,
                event.clientY - elementRect.top,
            );
            theContext.stroke();
        }
    };

    reset = () => {
        const { theContext } = this.state;
        theContext.clearRect(
            0,
            0,
            theContext.canvas.width,
            theContext.canvas.height,
        );
    };

    save = () => {
        let { drawnFiles } = this.state;
        const { theContext } = this.state;
        drawnFiles++;
        let image_data = theContext.getImageData(
            0,
            0,
            theContext.canvas.width,
            theContext.canvas.height,
            {
                colorSpace: "srgb"
            }
        ).data;
        const canvasComponent = this.canvasRef.current;
        const dataURL = canvasComponent.toDataURL();
        const CustomFile = {}
        CustomFile.dataUrl = dataURL;
        CustomFile.name = 'img' + drawnFiles + '.png';
        CustomFile.type = 'image/png';
        srcToFile(CustomFile.dataUrl, CustomFile.name, CustomFile.type).then(file => {
            this.props.updateFiles([file]);
            this.props.increaseDrawnFiles();
            this.setState({ isSaving: false });
        });
        this.setState({ drawnFiles });
    };

    changeSize = event => {
        this.setState({ size: event.target.value });
    };

    render() {
        const { size } = this.state;
        return (
            <div className={classes.wrapper}>
                <div className={classes.controls}>
                    <div className={classes.size}>
                        <label htmlFor="pen-size">Size</label>
                        <input
                            type="range"
                            id="pen-size"
                            min="1"
                            max="15"
                            value={size}
                            onChange={this.changeSize}
                        />
                    </div>
                    <div className={classes.actions}>
                        <Button onClick={this.reset}>Reset</Button>
                        <Button onClick={this.save}>Save</Button>
                    </div>
                </div>
                <div className={classes.canvasWrapper}>
                    <canvas
                        ref={this.canvasRef}
                        onMouseDown={this.start}
                        onMouseUp={this.end}
                        onMouseMove={this.move}></canvas>
                </div>
            </div>
        );
    }
}

export default PenEditor;
