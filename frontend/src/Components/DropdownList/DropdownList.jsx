import React from 'react';
import classes from './DropdownList.scss';

class DropdownList extends React.Component {
    constructor(props) {
        super(props);
    }
    render() {
        const { className } = this.props;
        return (
            <div className={`${classes.wrapper} ${className ?? ''}`}>
                <select>
                    <option>Variant1</option>
                    <option>Variant2</option>
                    <option>Variant3</option>
                </select>
            </div>
        );
    }
}

export default DropdownList;
